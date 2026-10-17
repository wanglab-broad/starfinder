"""GA1 (b): golden digests of the six accepted context presets (docs/synthetic-real-appearance-validation.md, GA1).

The real-appearance scene is opt-in (E1 of docs/synthetic-real-appearance.md): a scene without its settings is exactly
the accepted cellular scene, and the six ``context-*-v1`` presets keep their outputs. This file pins what
``save_context_scene`` writes for each of them, as generated at the base revision of the implementation batch
(8bf9a5b, before W-516's first change), so that a change of any generated value of an accepted preset fails here.
Digests are of content, by group of files, as the golden test of the sample presets computes them
(``test_synthetic_sample_golden.py``):

* CSV files as written, each top-level table its own group;
* TIFF files as their pixel arrays (dtype, shape and bytes), because the file bytes also hold the writer's version
  string: the sample labels, the FOV images, labels and tables, the reference-round stains (``fovs/<fov>/stains/``)
  and the morphology round (``fovs/<fov>/morphology.ome.tif``) as groups;
* the JSON records (``sample.json``, ``context.json``, each ``provenance.json``) as canonical JSON without
  ``numpy_version`` and without the ``files`` map of ``sample.json`` and ``context.json`` (the SHA-256 of the files
  as written); the array digests they hold stay in.

Repeatability is that of the formed model (docs/synthetic-specification.md, "Random streams"): exact for a pinned NumPy
build on one CPU. The three routine presets run in the routine tier, the three ``slow`` presets in the ``slow`` tier.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.synthetic import context_scene_preset, generate_context_scene, save_context_scene

pytestmark = [pytest.mark.synthetic, pytest.mark.golden]

# Taken at 8bf9a5b (clean worktree) in dev-s210-export-20261007T050852Z (NumPy 2.2.6), one CPU, one thread, by the
# W-516 run's scripts/take_context_pins.py, each preset in its own process.
PINNED = {
    "context-dense3d-v1": {
        "bright_regions.csv": "bef31f664ad3b306149169a37e656ed3693b7af88fd36ec41eca605a4fb0f69c",
        "cell_context.csv": "accafb5f14ae48381800d2cb2f539c7caa1c7de2d7cf3d1c6362989291893c29",
        "cell_fovs.csv": "ba5985ca2d71cbc0dce56468a4be339a48125441e94edde72d6f6136caa150ad",
        "cell_gene_counts.csv": "f6586a56ffcb67b5242d8ddabfd8c51c125df377d8cc807eba1d278e38bc6ca2",
        "cells.csv": "53493ed5b92560596eb7a3b91ed5b5d542a0e6d735d4cb2d47424c38e9e012f4",
        "context stains": "84390fe6bcf56970389462f8b73a795d68e9ad5557ff297170ee61ba26603169",
        "fov images": "8208a30c19605a169275b6d1a3cec81ec3923b89f41ab6522471b5fdb7bfa6b0",
        "fov labels": "371d29f5417c599ddee21344f1a4aad11a3f3558b54d8745414cbe1248f81b51",
        "fov tables": "13d2b89bee04f9aefc046a26bac6b4f223ee7500778bcba83af4bf4f7db4e548",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "868afd6c2914ee6133e2b1db134481ea94f879c88f3ff9c9c776f37064604488",
        "marker_pairs.csv": "735e0b715c6032b954db7c186635567145e8b9e9b9434a03491616b7f00a2572",
        "molecule_fovs.csv": "7c7de57ada958271750ae32cbbe2e20bb464035c61095d1c766f6761dde56244",
        "molecules.csv": "8fcea7b95f23ef9ba6df9ce8a3862e7cb315189b14d47820f6853c44b43125e9",
        "morphology_truth.csv": "74dfb2c1d9bb790025b572f78f3ec0bf926087e889c1425c9e4b9775297c31c0",
        "nuclei.csv": "9092664ff1519a18f85b88266459a89020a0d0dcea518ea5c65db2bccc91a05d",
        "placement_truth.csv": "f79bd2e6f892855e8f69c457e645cc7c8a2b02b2b7e4978d5b2c4084393b5d25",
        "positions.csv": "ffec325dcc6fc082202f4baaccddfb7eb01bb64dbc1a3c8a485d39df1e117951",
        "records": "403198f28bf8a73f22b1b2d282a10378c5497f2052c4c285d42a9173bb55e899",
        "regions.csv": "bd7fddadaf4ac142b418216e205fd6e649f1c00c37a3eb6882196b743296c8ed",
        "sample labels": "f60df984b2ae3a31a10956808ff956145a59c70120162af3071dbd286398818b",
        "stain_channels.csv": "b982fdabfa3b463f6dd8df1f11756c4a7f318f140290dd41a7ba173845e3e3a1",
        "stain_objects.csv": "bb12172b657eb921d72834a86aecd5b5258c364da5aed1f56056c46c7777d835",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "76c85129c9225e244c2e521271de497b37478f7d1da3e6d176ffc1aac9d10fc0",
    },
    "context-tissue2d-v1": {
        "bright_regions.csv": "7a1d08ed935e02d7a8f900648e2c1998faa6fd4367c37c4de067f092ab36d15a",
        "cell_context.csv": "6dbc40f624a18da50d040bb85daefda428339fe58141e5a79d1e5fd5ed30c67f",
        "cell_fovs.csv": "9e1550066ea4f9568faa1ce468229848d00c038d180a535212743eef6cb4dc90",
        "cell_gene_counts.csv": "9acd8643600df2af234e4f42285daaa0bfd069d7c66922b5f411022929108892",
        "cells.csv": "4e848dae81e28fa6be8b0e540b30bd0f50dd3d65cb0c60f757a3155fc0f0dbe1",
        "context stains": "e0e474d7171fa887f0efb5bbee2e5e37957685c64f893a31028062b8ec26d560",
        "fov images": "bf5a5237a8639111a3d3d7fc7b8fa657cde139c36ef6ffb42cb009c18f27bf78",
        "fov labels": "3feedb067f1a9699212f13313cc339447f7a117d090befc7d3e4bb82f42d3245",
        "fov tables": "c7e1fec7030eb845c1c8c9cddd2665071c871e7d182bab267c1341ab5b8fc4e7",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "f93e3519e7821616bb53fe6b38504b6851deefe88eba96f2d476942eb5e5f964",
        "marker_pairs.csv": "3874a57e2ad45c6b22582adad9a21dd8fbf9ba2262d77679e537cac6c71690c8",
        "molecule_fovs.csv": "a1e856af82c6aca3f756d40bd7f7f55b7aaad59cac4082c7044e7ba6a3cf22dd",
        "molecules.csv": "37c9dd47b86ca74195217084ff83d560b03e22680dc99bee4cd3ab027927529b",
        "morphology_truth.csv": "74dfb2c1d9bb790025b572f78f3ec0bf926087e889c1425c9e4b9775297c31c0",
        "nuclei.csv": "72e6dd5272a4ed9cd929583811cd896f62fa0392e5a754c8b6c2089ed5a77f61",
        "placement_truth.csv": "893798c2d480df2e40b30cc2cdd374879ec47439bb8b8b98ba125d6d227e9db6",
        "positions.csv": "e0d221bc896a133bfc37316046c2e04ba87e6e0a2a3d276cc00a19e83042c96a",
        "records": "8b4a31833749ccdfdb6e5aee5714304e1a301b4e33cc10ec637b871949b35f78",
        "regions.csv": "c067f91e8ae62328b6a2e6408b23f83952215798ef0f3aaa0bf54cba9cd72167",
        "sample labels": "8b6f148d04fc376e3138f978354bc30f105033a8dbfb03f73684d6fba891fdbc",
        "stain_channels.csv": "5465efa3814aae35116dea629586f1325810ea49485f660bccef577df6c5d95b",
        "stain_objects.csv": "b340c32bbb2b1084c1bf0268725ea56926c6455afe4477f1ea8075b6a89173f4",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "4007b7892503cb2cc4d84a7c75195d0a1125bcb46bde716e3d132aa6fd82f98f",
    },
    "context-culture-v1": {
        "bright_regions.csv": "deecbb3b64508f1641ae3cb709aa0c627bb778cb8c6d17c63c66138339a479e6",
        "cell_context.csv": "75293613cbd471bd78f89bba28d35c022b7be67ab2ccefdaf1cdb568735a36ff",
        "cell_fovs.csv": "668291aef331f3188d204e8bf65733f30aaf279fe8fbb654c22b5a5277535df4",
        "cell_gene_counts.csv": "833470fc66e25e0cbf8776cc8c64eb5b24f5037a56fc1ecb41b9c4daa160ddd1",
        "cells.csv": "d72edd0db2affc60934e1d074c2331b33de99858227548ae247c53b4ed39255b",
        "context stains": "61a70c030f99d9b2bfe0662b5e5b114e2a981071a8fa6582913444ffac5a791b",
        "fov images": "e700c1986627abf5cf9a72b71e03318893e323179c747cf20dee401e4cf0a4bb",
        "fov labels": "1531b2a97bc18491a321c4ad29cebe61b072494e4215eb8a5e2f3b3ab505ecab",
        "fov tables": "04040119d1d17742ae55b5c558d2b960ca9f306f39e223bcfca2cb6144af040e",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "e5974173ada8acc80274d16f802db2feb36d8655c6aeeb7c05135dd258f2abfe",
        "marker_pairs.csv": "7b03e5988de14ab4b2d34baf3a59aeb0d87980f3f4dad86db906743348ab4244",
        "molecule_fovs.csv": "f7df069141d9ced75d99d29aa2382c02709b021900e478a82dbc988148e3d7f5",
        "molecules.csv": "bbc4d6dd1d65502749cfe8e1c598cac05db8bae81bda19a3751a035b8f0032d6",
        "morphology": "11dbd8c22af2d4755df46a7a71374cf1ca859d9804edf15c737b9ca09bbf6081",
        "morphology_truth.csv": "f85d560416370d62efee304fd1e483e1114f4c5543ebb5c1739307c3913d4578",
        "nuclei.csv": "7294f2e1ffda7c6f0d308ac302e3b8e0fc32b03ab7e4afd0db6740636afdc401",
        "placement_truth.csv": "d8cd356346a6b61d716d538bcc52791b8cd17560e0f3a089d7b3a6777b75db24",
        "positions.csv": "a3cb5a12e8c84371bebc7ede3eb9e5816be4399c361885870012a44c42e39caa",
        "records": "f00e86b189dbf99ad829241da961c9933c086a253b9befb852e2e77da7e5df4c",
        "regions.csv": "7146e9ccdef0c1666175a9d4e7e0006d8ef13f7c9651dafcddada69b784bcb5b",
        "sample labels": "85706dc7674c16229ce883be5c9c6c12bde31da9e91209aebd3d38e1621ae342",
        "stain_channels.csv": "99c5041a89d66235f5e571e22a87733c91d537aa55fa55f7d005e653748ad8a8",
        "stain_objects.csv": "72d5bcb3eb5139749b84b4ede0c96c45e251672081948e629a7248c277dc9841",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "76414f5cae573a60598f3a0062c87db9eaf9057fe5b8961d0e09baa695a58d85",
    },
    "context-dense3d-slow-v1": {
        "bright_regions.csv": "8889e203562492d8a1cc2bb2c18c35ee1ceca49020a7be7dc5961565ff96508d",
        "cell_context.csv": "5088397d80170169a861d4280a82fd766939e2d5cf18161d278f51bed592307e",
        "cell_fovs.csv": "c59bcc0c5042655bdbac0284747880127f4c10d73afb6b3c19596376a236de89",
        "cell_gene_counts.csv": "c68c175c26506f2a08bd691720b62d07459a1e83297045d6ddec8b85ee1b5536",
        "cells.csv": "aca270262f132a9421e8679ef514c43eb4eeba96dc1dfc63e87ad4f92bb1829b",
        "context stains": "c665149ad9160b13e6bc7f530abf36e1ff705aaa4b30ab0c2a6fd7dd2177e3d1",
        "fov images": "29c5113bb4583be7b8a8dd4518330b11c5d3480f5342941c7dd5f7fdf898200c",
        "fov labels": "2ffe964c0fa3ab31cd6af0cecfa793a31a4a0c355f8199aeaf32da002d3cc6aa",
        "fov tables": "ad3523ef0512e4864059cc461ee029a94d73d2ca2a51aab89d90d18d2318bea4",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "78abe2a369ec6a097620500da5819d5286a00e91f8a95181089d52c376dfe41c",
        "marker_pairs.csv": "86767e3cef73ee9c3c7bef26cb1567d9a49e53bae1828d4897b3fc18920152d6",
        "molecule_fovs.csv": "74392aee1a17a205ecd45f08d1d0c6dbee4043b2d0c1f06333c0f3ba83889cb1",
        "molecules.csv": "09f7b2d528045e14379cf3e30e1d5551613d839423cb7f84e7aac1548735b946",
        "morphology_truth.csv": "74dfb2c1d9bb790025b572f78f3ec0bf926087e889c1425c9e4b9775297c31c0",
        "nuclei.csv": "8de2a3b9983ca1180c63b0fbad7827e5a1e565a94b894a4a47e2a2d6277d7c8c",
        "placement_truth.csv": "6ef134db138e54de93debe86db4c8cded63f0cd01df0647b6a7f15c27db7e491",
        "positions.csv": "cabf8800549b6458c0922907d4834fd1432f5dcf77976dcba38fa48ea8a40516",
        "records": "990834a1d1b6fa5379ffe278963e4263a5c51a0c6dff7c91f31e681320f2f369",
        "regions.csv": "d3643d710f8970a99aae9e1b66d51cd4d59f241bd4830b5cea04da86821ee034",
        "sample labels": "40352599da99ddb45e29a60114073291c65cde2602f565ed1e08392ff86e98de",
        "stain_channels.csv": "b982fdabfa3b463f6dd8df1f11756c4a7f318f140290dd41a7ba173845e3e3a1",
        "stain_objects.csv": "3101805f46a2d3f996dd275cb01d122d898497df09db8503b198c505e9605ab1",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "76c85129c9225e244c2e521271de497b37478f7d1da3e6d176ffc1aac9d10fc0",
    },
    "context-tissue2d-slow-v1": {
        "bright_regions.csv": "6d632d37fa258d2b477e802c300cc893c6f007b9d5fbd939198525a77062fccf",
        "cell_context.csv": "a0ee79cd958cf221e27d56025b35436689b8385884322d820423ff03b46d7aa3",
        "cell_fovs.csv": "08534883061768cf967329ee0c18e09b02b0e2077f905053b36d5586b189baf4",
        "cell_gene_counts.csv": "8f015e5a0d692af6d9f71a3ebc41858600d22a1a5280b1f4743c5cacd05d19e8",
        "cells.csv": "7d7b1961ba1eaa6ebb7b56346435e52c72b2468f20b0331c8b1b0d3938abc822",
        "context stains": "ffa2c1d83df12533710c0dc30db4e970a3896efce66b47f8ecb8fdc3755b2275",
        "fov images": "6e920e08ea867de08534eb0c0cb7b6d7e77dc80c304e993bd017aafeaa5b7d27",
        "fov labels": "ebdf0db65d548a8b8f03c6522d8b69c2dd0e5015b7366cad8ae53d6f254e7e63",
        "fov tables": "c770668bb32b1b932092f3a347f407bf97bd84dbab131812ae89d934ab455e69",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "db5ddf9424949fb8567fd5fd32ec382e88d6ca3ff529a3acf34a81537f91875e",
        "marker_pairs.csv": "780818bfcf40488b9b984b727134f4868020a5b80763cd1ddf817b0b746c30fe",
        "molecule_fovs.csv": "c34eb86652736c75fce940170415cac75977fe2a2199ec26e4d49fa55ae21569",
        "molecules.csv": "549bc4b3a829c479c294b492c4597a47968af0621fa6fdbf3bf40845dd530078",
        "morphology_truth.csv": "74dfb2c1d9bb790025b572f78f3ec0bf926087e889c1425c9e4b9775297c31c0",
        "nuclei.csv": "7f692e7773400c0547e11758c165279ee4361ad6be987d97f9847fd3147c524c",
        "placement_truth.csv": "ca32d6fe462a6bf1a1056a8e5ee00edfdb4c4a94838bb871c5ceb246063a6fa4",
        "positions.csv": "fa90a0e6b23d4981c25e9b073a82590332ce49fa91f62f17847eb6f4ff06b447",
        "records": "da0f9cc36ca2d6168a9b6b89c52d38b0340468f797b7849754c6eef86b59284f",
        "regions.csv": "ed0cae7fc4fe68e4837ccb9ce1a714560c0ec7e37b3c8845fec5e0b00522baf9",
        "sample labels": "2e65a2c2cce5e7bb76ecc9f1dafa4af142b8d2e704adf8497c0bec3466b6a869",
        "stain_channels.csv": "5465efa3814aae35116dea629586f1325810ea49485f660bccef577df6c5d95b",
        "stain_objects.csv": "b9ca65733c70f87707df8a96a55f8519abc0f5eb9ee29b7cfa97b51f27477973",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "69fda7d680bb5f4570be90c419d29b6244bb18a1fa1e5ccbd5fbea945f3fa040",
    },
    "context-culture-slow-v1": {
        "bright_regions.csv": "27687ee42393583df92944ae77fda0d3c15ed0b7074a8fd968e5c8158c186967",
        "cell_context.csv": "5327da422a57dcc0f63e94d25a635dd3e55718aed8133db56d8e7f735fabdf72",
        "cell_fovs.csv": "25219d8d5e160c5e9d4238c295e612e0bf8f326272335ed10b0527d5d21977c3",
        "cell_gene_counts.csv": "5b18cfa94fdde0f1fe8b3fd51d59b14df24eed4096f1eb017e460ebbcb2112ee",
        "cells.csv": "c55f39573c113f00c61cf9ebb11bbb4e2edbd96e14fe4dbeeec1bb83994c587b",
        "context stains": "4584e6cc1449444b711e99420f7edbedc3adf0a0f185d1a371e659ae9e52cd86",
        "fov images": "66be97a4644f6dd828ae9bb3b8e18ffff091b1ccd13ef37e75f279f4d8970adf",
        "fov labels": "ed21cdf525ec90d5b607f1945e9e39615b7f2978e27d2d71e897a03d0fc55f7f",
        "fov tables": "2f49943223f3d73f7941813e872ab305d37d1c7354cd1d6c1d4182ff8371dde9",
        "harmonics.csv": "a4824e7f9bd13f585e245b8fcebcff8782d01602fa26a5f28b62099b32c61c56",
        "lobes.csv": "640bfa2770cf35f7bed7cbeebab3f77b28eae9030967c7794c9c7bb81e5a2507",
        "marker_pairs.csv": "ea8b5c77fda441aa6a2bc83d9e6a77bc151af96c43e696f7014eb33229704ac2",
        "molecule_fovs.csv": "ec9d8228d869136f62ebfcbd46f4c9d98c3fadc14eb12fa813695e83f962f912",
        "molecules.csv": "bc19b00c894e239f54c2e733c48170a81b711d29a518123de7a0bbfec262715c",
        "morphology": "d848e1899d72eb7f1b3fbb555e3bb98fe932d66afe191fa07be46ca8702e8449",
        "morphology_truth.csv": "44a580dca93deb133bcb8bda9e2f2da0ff570fcd0cf61d143fb93ed0585921f8",
        "nuclei.csv": "5181f37b2693d10750244b21b9ab0caf99c6ba188e5deb930b7c7dfa11a0f83e",
        "placement_truth.csv": "95dacaac237700e9e1f87ad6d951ff1272e45be26a6a7e53ad23d4b17150e25b",
        "positions.csv": "bd0dc95c1c10b5e3366a8a054483319adfe423dfef6715b567d005b222d5e99f",
        "records": "1cbd5a1140b73399b47322d27652fb5e816e8bb338dd6eb1ce53f16f08f9491b",
        "regions.csv": "a2d9aba5fdc8e7b47e51d364147242584099bfdca3013036ecb6d84be08bd0b7",
        "sample labels": "c7315110f0e9d058cd23b7ebf86d3f69ca49025dfd764f10cc8f0f7f8c5b9c36",
        "stain_channels.csv": "99c5041a89d66235f5e571e22a87733c91d537aa55fa55f7d005e653748ad8a8",
        "stain_objects.csv": "69d6d7770d0b6198363443afcdb7cc6b02af2933c4a6b19882c0a7093d7c331c",
        "type_abundances.csv": "9966f3c88059edb2c27eba233dc40339afa8aaf0c011b12b6d07ed76c0d4ff0d",
        "unevenness.csv": "68495b36d84028d473a080835874a371f0aa7134ead1e5996e309c05e3c6de00",
    },
}


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k != "numpy_version"}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def _file_digest(root, name):
    path = root / name
    if name.endswith(".tif"):
        array = np.ascontiguousarray(tifffile.imread(path))
        payload = f"{array.dtype.str}{array.shape}".encode() + array.tobytes()
    elif name.endswith(".json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if name in ("sample.json", "context.json"):
            record.pop("files")
        payload = json.dumps(_canonical(record), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _group(name):
    if not name.startswith("fovs/"):
        if name.startswith("labels/"):
            return "sample labels"
        return "records" if name in ("sample.json", "context.json") else name
    if "/images/" in name:
        return "fov images"
    if "/labels/" in name:
        return "fov labels"
    if "/stains/" in name:
        return "context stains"
    if name.endswith("morphology.ome.tif"):
        return "morphology"
    if name.endswith("stain.ome.tif"):
        return "fov stains"
    return "records" if name.endswith("provenance.json") else "fov tables"


def saved_digests(root: Path) -> dict[str, str]:
    """One SHA-256 per group, over the sorted (file, content digest) pairs of that group."""
    members = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = path.relative_to(root).as_posix()
        members.setdefault(_group(name), []).append(f"{name}\0{_file_digest(root, name)}\n")
    return {group: hashlib.sha256("".join(lines).encode()).hexdigest() for group, lines in members.items()}


@pytest.mark.parametrize("name", [
    "context-dense3d-v1",
    "context-tissue2d-v1",
    "context-culture-v1",
    pytest.param("context-dense3d-slow-v1", marks=pytest.mark.slow),
    pytest.param("context-tissue2d-slow-v1", marks=pytest.mark.slow),
    pytest.param("context-culture-slow-v1", marks=pytest.mark.slow),
])
def test_saved_context_preset_outputs_equal_the_base_revision_pins(tmp_path, name):
    book, config = context_scene_preset(name)
    root = save_context_scene(generate_context_scene(book, config=config), tmp_path / name)
    digests = saved_digests(root)
    assert sorted(digests) == sorted(PINNED[name])
    changed = {group: digests[group] for group in PINNED[name] if digests[group] != PINNED[name][group]}
    assert not changed, f"{name}: groups whose saved content differs from the base-revision pins: {changed}"
