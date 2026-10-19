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
build on one CPU. NumPy's vectorised functions differ in the last bit between CPUs, so the floating-point values of the
CSV files and the JSON records are compared to 9 significant digits, with magnitudes below 1e-12 taken as 0
(amendment of 2026-10-19 to GA1: a GitHub runner differed from the lab machine only there). Integers, strings and
pixel arrays stay exact. The three routine presets run in the routine tier, the three ``slow`` presets in the ``slow``
tier.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.synthetic import context_scene_preset, generate_context_scene, save_context_scene

pytestmark = [pytest.mark.synthetic, pytest.mark.golden]

# Taken at 8bf9a5b (clean worktree) in dev-s210-export-20261007T050852Z (NumPy 2.2.6), one CPU, one thread, by the
# W-516 run's scripts/take_context_pins.py, each preset in its own process. Taken again on 2026-10-19 at a9e7b59 with
# the floats to 9 significant digits (GA1's amendment of 2026-10-19): at a9e7b59 every group still equalled the exact
# pins of 8bf9a5b on the lab machine, and the new pins are the same with NumPy's AVX-512 paths turned off.
PINNED = {
    "context-dense3d-v1": {
        "bright_regions.csv": "5dbf2cb3143bdcee9b05e109a7c79583c7f0b86171661a60274a35d835e42ac2",
        "cell_context.csv": "21c014e31a02c0566ce64e89180a65a308d24e13ca5b6b6492cfc31181304b71",
        "cell_fovs.csv": "f24061cdbdea5b5aca6e60127c047102218f6595c68b573c803379cedbb99ca9",
        "cell_gene_counts.csv": "8203ad72379314ee88e094609dc78270da01766edd54d16ff99628207cad6e9d",
        "cells.csv": "5f9236b517c7cfc5016e0bca5e10a2cc43eddec39d40ce02293266a41d8cee43",
        "context stains": "84390fe6bcf56970389462f8b73a795d68e9ad5557ff297170ee61ba26603169",
        "fov images": "8208a30c19605a169275b6d1a3cec81ec3923b89f41ab6522471b5fdb7bfa6b0",
        "fov labels": "371d29f5417c599ddee21344f1a4aad11a3f3558b54d8745414cbe1248f81b51",
        "fov tables": "671274102f3eb64733ec41f1160a43b765f9d2ee64b46a92e55ad348cd9d337e",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "acd38411f7ee3c88eb5037a7d77ca9199cd23694e7d390901715ea7eee34df48",
        "marker_pairs.csv": "72d74b6d1f93c8f078ffe9b99a95e3b3f15b8a563b490b8af3bb1b0f756aedaf",
        "molecule_fovs.csv": "e0b424193cbc2b4e96e6a11fcf0791c59927cd1790533c28d79ccc09f7b4db20",
        "molecules.csv": "c359f9b00b1648ca91fba8a0cbe7c3c912a389928e3394d4dc2a165bb0ab4a9e",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "984a768d07824c8b3136cc739181c8ec7464e0cfdffd370b96030898c7aba84d",
        "placement_truth.csv": "ef9dbba09522f0fc73d8320544ac93974f7afce5469fd5a05c93caf85d05458c",
        "positions.csv": "3606a86a44b9236161f0113c832908e838e9b1e42bf56219a20db64335927c0a",
        "records": "1988ff4b740bc1ed1b107b13dd41c128f8522ff633663484e2e8cea0774b7938",
        "regions.csv": "841067af46044cd19b0ce99ec618d332081e1477f71db989bc0c871d2fc90979",
        "sample labels": "f60df984b2ae3a31a10956808ff956145a59c70120162af3071dbd286398818b",
        "stain_channels.csv": "57eac57b213298e95a531be2b17336e17a65b956df0df92f0d0ec4750a7dee1a",
        "stain_objects.csv": "4eb312252bbd7283a0c0b8127ed8e34438e3d95960f75f3cd5e298bb447d0c63",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "74e4a3df22e81ab74e519ab2870233a5839238e6727099253fe886437f2e0002",
    },
    "context-tissue2d-v1": {
        "bright_regions.csv": "64f34c5d17e25d9edb6fd3a72230871ffc74fcd02eb421239a884aa62ab8975a",
        "cell_context.csv": "b993a03444a5f4a77a50724ebfb4b6c2d3fe88c55845b2622c1d7acd293277d4",
        "cell_fovs.csv": "9477d4692f26290fcaedbd794d23f52a1cf3e2b850184aa139dbf24bae0f9567",
        "cell_gene_counts.csv": "10609016ae6e09f257e2e72dbf8362c5c37c4f3042d83e7e1d24cb0926743464",
        "cells.csv": "f96f699a7073beb95d487978f803ac4c3010c96d7e9b467b5cfddf46147c63af",
        "context stains": "e0e474d7171fa887f0efb5bbee2e5e37957685c64f893a31028062b8ec26d560",
        "fov images": "bf5a5237a8639111a3d3d7fc7b8fa657cde139c36ef6ffb42cb009c18f27bf78",
        "fov labels": "3feedb067f1a9699212f13313cc339447f7a117d090befc7d3e4bb82f42d3245",
        "fov tables": "cd339cc1a2cc390ba74393909cc2377a3096af059407694c311f292a09f9d204",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "8002e4fd741d45f2b6535ee912a6d97eca99bde59da48c8b26bc99d418883417",
        "marker_pairs.csv": "179daab6d78637862cb9c993a9c247bb3bce9658d2d6bb3e127d99f54d7b4a88",
        "molecule_fovs.csv": "d647f51509958686c879872550fc3b5ca6d648e182ac6b93092cc05821f718c5",
        "molecules.csv": "2a978cfe5186286d9bc71695daa03486b97b90c2c450c2acb93d8941bbc2c1c7",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "1e72511cee6c0075c5bcf1d836a2c923ad847d97594dd23f63fc74fa3748e68d",
        "placement_truth.csv": "4720c3109b07d39602c62a8a02a131aa86ed8320a5e56eeddaf61b2386595458",
        "positions.csv": "465ba0d518c8f35d9ea968e7089747b8c81a1a9a0d570493b8c2045cde3e8c2a",
        "records": "e60118a3601524ac45db91e080c410c1671e370f2e58ab4c842e96a15d5b7f84",
        "regions.csv": "1d6798a37f4b26e1038a9e415ec8b30c17c2f3f4b63b362b8d942f12a082949a",
        "sample labels": "8b6f148d04fc376e3138f978354bc30f105033a8dbfb03f73684d6fba891fdbc",
        "stain_channels.csv": "b0edcde7497c38f742f17ddacfe4cf6aa498452517046f55ac2cc94f86850e5e",
        "stain_objects.csv": "9742c21a042fad11b9d0987b742eca2b0e91b82d1a44f5b29426dd519e9bd8c1",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "bca4ef83ba9365c70bcf0d67a91e5d343eaf3551a999981d54c5b7179b26cc69",
    },
    "context-culture-v1": {
        "bright_regions.csv": "a1b207a1dd907c37923c8486d1e7fa26dc566772acd20092a3b793a951a1691b",
        "cell_context.csv": "b264173cce41b04ca16688455edc0833282ab335dee43069771df7f3df415799",
        "cell_fovs.csv": "c5a13ca528dc318d554b538a155a6b5493a11ed9c029ac844c8c320a9dcd5de5",
        "cell_gene_counts.csv": "60c3e7bd2595b60c8551aed4869a7c1584c5d2fa837a558367c9e86d9e476d57",
        "cells.csv": "17c8c4ed54a777a97a88a3fcb69fcbee5fb3b51e201e8f7c285a0e7ae702ddc3",
        "context stains": "61a70c030f99d9b2bfe0662b5e5b114e2a981071a8fa6582913444ffac5a791b",
        "fov images": "e700c1986627abf5cf9a72b71e03318893e323179c747cf20dee401e4cf0a4bb",
        "fov labels": "1531b2a97bc18491a321c4ad29cebe61b072494e4215eb8a5e2f3b3ab505ecab",
        "fov tables": "044bd99048af5a8b23ddf2dfb5ea254d2f36c58809db034bc9e85a8d1cfeb958",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "2df8a30a4cc95187f058666181c80c05d5e149bfab78dbf3b88c87e8c4cb8f64",
        "marker_pairs.csv": "b716e0aa20e63b7da7359e9dbc40131fc266e06fbdd944d45ca37b158923d358",
        "molecule_fovs.csv": "1a6e0c207809104b831b4adc675a93eb90c23ba790d1bac1677d3820bc6007cd",
        "molecules.csv": "76defec7d9266985b6052b707aae449a95560eb1f2bbb2fa0efef94b269d3438",
        "morphology": "11dbd8c22af2d4755df46a7a71374cf1ca859d9804edf15c737b9ca09bbf6081",
        "morphology_truth.csv": "36a5dd23545cf2799386a2df2974dbfc8d40780cffd04a7b9dbeea46e1b360fd",
        "nuclei.csv": "c110f1fbcddbe7f7afd4fd3bdfa2c47db4add61dbd73aefb2319826d81a250d4",
        "placement_truth.csv": "0810bc6589222aeb961bfd06d9c563dffb4d41fe8c86eb8ac85a6dcedd6e5e7a",
        "positions.csv": "d4e02f396011035345316c145a54d9e92239139643aaaf490eb13e06545fce69",
        "records": "f1ca868aa21259a322953e2b2c88755cf23c3d1db7ab1bfa7cc11aac240e0359",
        "regions.csv": "d90a859c46a966ba56131212c45b22dcf867506209d84c0f52fa9df63e3361df",
        "sample labels": "85706dc7674c16229ce883be5c9c6c12bde31da9e91209aebd3d38e1621ae342",
        "stain_channels.csv": "5d81a7f67b7f61643dfb0adc5c0b2211e7b139baf5241c84dfa692212891d520",
        "stain_objects.csv": "402d454d48ad72b4728b4a550be07d3962ad942641d44b2cd24ab6aa59714202",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "25339c2502c8a4bc62d7e7245ea2044807a536a12d271eb5e5e29449adac756b",
    },
    "context-dense3d-slow-v1": {
        "bright_regions.csv": "aa9c37f34e7b90fcdc124a6645c9e736ec9ca3f4e2fb17e84c4bed90da4aee96",
        "cell_context.csv": "b9a2394b35504de16ec0f03c8e6a0204576c2bc0931b083b8538a06a1c373425",
        "cell_fovs.csv": "03c698fea72a15caedc320bd1c9cafe049f14da1fa16d2552b11cddd9bd9cc1f",
        "cell_gene_counts.csv": "62087b31609fbbc37214332feea3bfa617b63cd6c47e04450f51a49f90f5b3bf",
        "cells.csv": "8fb691a4a3b01157676c1aacddb1860b5134c84657fcfe4c0a884a4344b32ccc",
        "context stains": "c665149ad9160b13e6bc7f530abf36e1ff705aaa4b30ab0c2a6fd7dd2177e3d1",
        "fov images": "29c5113bb4583be7b8a8dd4518330b11c5d3480f5342941c7dd5f7fdf898200c",
        "fov labels": "2ffe964c0fa3ab31cd6af0cecfa793a31a4a0c355f8199aeaf32da002d3cc6aa",
        "fov tables": "cf843f7724e1d47d3848ce3a3cf7a15538e32ae248cd43bc3d39b7a155267e73",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "4501b75a0f23e1f83a1cade7c1eca45d368def09295c77d60ac52396045117e8",
        "marker_pairs.csv": "a734163ab9dd4074592b3bb717b487e24b55c1128d8fb138f132c9b4b59509a6",
        "molecule_fovs.csv": "a404962d323b4e4193093d96ef51b6163ebbb3834a342696e40f5981f19027cd",
        "molecules.csv": "d31fa26aed8de2fdfe6dd8fbce31608423b6eb1aa046e694eff0132cac9e296c",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "207010d0ac0d91cd09efb8a3bf9385ea70b58a1e78bc6ff0658d3d5841d10bb3",
        "placement_truth.csv": "f280758a39cbb33a93bf566634d2ac66a8ab073cc9106ecc08ff05faaf32e75b",
        "positions.csv": "ba323317f14a495bb4a2bfeabf72ce489aad5044d59123bf46813acf6e80ddab",
        "records": "9f99e29ebb06f87a847c787b358b44121c3cd7f9399bc926ca0aac89c1f9e861",
        "regions.csv": "b67c0eb8e62104cad0aebc4db0e437b1732366b624b0726d851f46a02385d037",
        "sample labels": "40352599da99ddb45e29a60114073291c65cde2602f565ed1e08392ff86e98de",
        "stain_channels.csv": "57eac57b213298e95a531be2b17336e17a65b956df0df92f0d0ec4750a7dee1a",
        "stain_objects.csv": "22995bd2aebfd021b2c2e7802f30d78237f66ce72c27c421fc6f22a206a24aef",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "74e4a3df22e81ab74e519ab2870233a5839238e6727099253fe886437f2e0002",
    },
    "context-tissue2d-slow-v1": {
        "bright_regions.csv": "efc0c878c0bd30986d3d805ef0d4c6f26fa53cc9a1791bf27191e5c3f427918f",
        "cell_context.csv": "df4956cdd3c06b6305b37b093e039fd26b508e85a79168556a6cabbd1b49844d",
        "cell_fovs.csv": "a248b5690c9ffb17708ffe7a79cd955937ed101b57d8e6fa837011af7e7eca98",
        "cell_gene_counts.csv": "67a1025437a1dbbe41898c82ebb2afc7b0c7540fbec6fb2124a4b3489c3e911f",
        "cells.csv": "2a3492786101385e41043d24643ec309b08ea4ad47d57386e5079fa80c914afd",
        "context stains": "ffa2c1d83df12533710c0dc30db4e970a3896efce66b47f8ecb8fdc3755b2275",
        "fov images": "6e920e08ea867de08534eb0c0cb7b6d7e77dc80c304e993bd017aafeaa5b7d27",
        "fov labels": "ebdf0db65d548a8b8f03c6522d8b69c2dd0e5015b7366cad8ae53d6f254e7e63",
        "fov tables": "3232b56ef41bc693ccf2ea0cdc8428d77266a2c224cc65979f3ac2a411092dab",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "d789741f393fc4cb744f9b8a103f194b20740096833239e92b291791db829304",
        "marker_pairs.csv": "7ca9789d3c95094b66243b46f3acfca645a133b43652333b529dbd3081a56716",
        "molecule_fovs.csv": "9b48528e7784da2b39d20a44176837df3b28cd40c873b14bed477cc09745895e",
        "molecules.csv": "42446f2522c72dbe72e81b78fe2a363327533aa9bef9e36a0198b5388b185026",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "dff8e7080c355366f36e46dad8a2f4b616b4a10c176d20ba42ee9133228f3bd7",
        "placement_truth.csv": "3aa30a7a3f322de6a9af7de1c8e789975a7dbd3f8a716cdfeaac0ab613f9b0f7",
        "positions.csv": "88153c3ac42e3132dc830b5e94e45e696c1fe2bc2306014d42355216ac79c690",
        "records": "a352c6e81f1aacfa763f09d32572e6c8cab050f6fe4696e87b9f265a110b7c55",
        "regions.csv": "a429050cece1a83556329cb97f7dffb962c7f594fb2b2bab5924bd4615f778b8",
        "sample labels": "2e65a2c2cce5e7bb76ecc9f1dafa4af142b8d2e704adf8497c0bec3466b6a869",
        "stain_channels.csv": "b0edcde7497c38f742f17ddacfe4cf6aa498452517046f55ac2cc94f86850e5e",
        "stain_objects.csv": "8905576050507c0c9aa4885d5e51bf5af4bb7bdc39a7610761bfaf48636a2223",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "46295ab5267d81ca490c4b18867f51c4413de464ebbc179b1888b532da6626cc",
    },
    "context-culture-slow-v1": {
        "bright_regions.csv": "5f6c37a85447ddb741f3ac7773cbe1956a292415266a94934f3b6320fe2c3fd7",
        "cell_context.csv": "0d3315a7a35cf4f6b4527294436d70efe6b4f1e82dce67a79e3b1666387abb68",
        "cell_fovs.csv": "b8f25bfb1d6946dc55f9e2306ee32f10584a1182bd1f3bf60bbd75c09e4eb4c6",
        "cell_gene_counts.csv": "eb547aef70fbad7c9a19548121c5b650603d2cc98273e23055a108af0945291c",
        "cells.csv": "f2f962f774cc2f0239b7ac0887f86aadd648943b8a38b730af348258db1150de",
        "context stains": "4584e6cc1449444b711e99420f7edbedc3adf0a0f185d1a371e659ae9e52cd86",
        "fov images": "66be97a4644f6dd828ae9bb3b8e18ffff091b1ccd13ef37e75f279f4d8970adf",
        "fov labels": "ed21cdf525ec90d5b607f1945e9e39615b7f2978e27d2d71e897a03d0fc55f7f",
        "fov tables": "b97f09401196ecd277ab6343090088825abf9687190a7b018805b7d54594245f",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "99b5e8bfad0e4724ead146e665db15846d8511713c3f6d0097934e74dbb871b8",
        "marker_pairs.csv": "d34f81e795f807b75afd02a95135ca5b2c1ce5df57b826101aff24e01d10ff16",
        "molecule_fovs.csv": "96316357f9dd12e1da0f7c1b523076ea277c8934595e699dd83ba9cd6c810378",
        "molecules.csv": "cf2099b68ed9c6b54f20042b6b5849f27960a3cb4a417e45e381daf371c5b545",
        "morphology": "d848e1899d72eb7f1b3fbb555e3bb98fe932d66afe191fa07be46ca8702e8449",
        "morphology_truth.csv": "55ee138e296e8ad2d1f6e9756a2276518758d7af305cfbeb678f352d3ddb3d04",
        "nuclei.csv": "5ec72505d9438cf82b074e099e5528a0a2d5881af51abe07e41fd43f7fb1d1bb",
        "placement_truth.csv": "91befb4c9ddd19a2cc6e0f3a202c94c973445028228e60218154c121717bfd9e",
        "positions.csv": "74758b503b259d9d1f2da0e8557a3de917b4af09c4b17f9464a4e22630542c73",
        "records": "3ee1294f84beef3b32c9f2662c40d365fcf4e7e343cf6c35e4433d3930708b8a",
        "regions.csv": "24c6874049ef5333decf0cb2cdafc405ceecffab9b1004bde0990eb30e430b81",
        "sample labels": "c7315110f0e9d058cd23b7ebf86d3f69ca49025dfd764f10cc8f0f7f8c5b9c36",
        "stain_channels.csv": "5d81a7f67b7f61643dfb0adc5c0b2211e7b139baf5241c84dfa692212891d520",
        "stain_objects.csv": "fad225a903e7316d3b7c5ceb6f3523561083ba00e31bc3bae819a9e07250566e",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "9993f80f415df01ac0fc83b2d00f9a0db2a544dbb94b6b7b196e9f20d958c624",
    },
}


def _number(value):
    """A float to 9 significant digits; magnitudes below 1e-12 are 0 (the last bit differs between CPUs)."""
    return 0.0 if abs(value) < 1e-12 else float(f"{value:.9g}")


def _canonical(value):
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k != "numpy_version"}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return _number(value)
    return value


def _cell(text):
    if not any(c in text for c in ".eE"):
        return text
    try:
        return repr(_number(float(text)))
    except ValueError:
        return text


def _csv_payload(path):
    rows = csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline=""))
    return "\n".join("\x1f".join(_cell(cell) for cell in row) for row in rows).encode()


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
    elif name.endswith(".csv"):
        payload = _csv_payload(path)
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
