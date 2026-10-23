"""GA1 (b): golden digests of the six accepted context presets (docs/synthetic-real-appearance-validation.md, GA1).

The real-appearance scene is opt-in (E1 of docs/synthetic-real-appearance.md): a scene without its settings is exactly
the accepted cellular scene, and the six ``context-*-v1`` presets keep their outputs. This file pins what
``save_cellular_scene`` writes for each of them, as generated at the base revision of the implementation batch
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
CSV files and the JSON records are compared to 6 significant digits, with magnitudes below 1e-9 taken as 0
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

from starfinder.synthetic import cellular_scene_preset, generate_cellular_scene, save_cellular_scene

pytestmark = [pytest.mark.synthetic, pytest.mark.golden]

# Taken at 8bf9a5b (clean worktree) in dev-s210-export-20261007T050852Z (NumPy 2.2.6), one CPU, one thread, by the
# W-516 run's scripts/take_context_pins.py, each preset in its own process. Taken again on 2026-10-19 at 8606450 with
# the floats to 6 significant digits (GA1's amendment of 2026-10-19): there every group still equalled the exact pins
# of 8bf9a5b on the lab machine, and the new pins are the same with NumPy's AVX-512 paths turned off.
# The "records" pins were taken again on 2026-10-23 for W-559 (the renamed "type" and "generator" labels), after a
# saved comparison showed the base revision's records (89ff80f) equal to the new ones once the rename table is applied.
PINNED = {
    "context-dense3d-v1": {
        "bright_regions.csv": "c4a281ab0fefd3a3562936a99b5280395e362b124361c772adcf100703527cb6",
        "cell_context.csv": "e4134b5f6ef80db7c56003537bef64c61d7ca1be34d47be637d04d1f8e290150",
        "cell_fovs.csv": "f24061cdbdea5b5aca6e60127c047102218f6595c68b573c803379cedbb99ca9",
        "cell_gene_counts.csv": "8203ad72379314ee88e094609dc78270da01766edd54d16ff99628207cad6e9d",
        "cells.csv": "de3658a55bbb6e263b81a7641ef76e8c4b0370fcc4862550dd282918ce531fbc",
        "context stains": "84390fe6bcf56970389462f8b73a795d68e9ad5557ff297170ee61ba26603169",
        "fov images": "8208a30c19605a169275b6d1a3cec81ec3923b89f41ab6522471b5fdb7bfa6b0",
        "fov labels": "371d29f5417c599ddee21344f1a4aad11a3f3558b54d8745414cbe1248f81b51",
        "fov tables": "e8642788a8ea7648027d74ed3d267ec89c8488c40423e9d1a95915075647702e",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "1222a90c9425a33514deeb36d5617f79fd911043bbd606613c06574913ef1957",
        "marker_pairs.csv": "72d74b6d1f93c8f078ffe9b99a95e3b3f15b8a563b490b8af3bb1b0f756aedaf",
        "molecule_fovs.csv": "f23be38552ce7a88bc198c387fb227b35474793188e127cd2fb5bc1c591837ac",
        "molecules.csv": "57623c8752ef519296af504d4cdc6f777ce10dd2d43e23bbaf953620e375377d",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "92d353f6c431a15f64537a81f64cc7b0dd34306a88d5ff25105feaa72a052609",
        "placement_truth.csv": "7ed8ace5da46936f26fc8ae421b8294a182bcab99762dfef95d2a0f7490c31b9",
        "positions.csv": "72a704e59318e53e647115a1ba316d7f4fcac369a6203f43fc0c2bd773d82cc7",
        "records": "8f8392a23622d86b688be68c240f6b29b924396b9e019bb64740fa78d83b1d55",
        "regions.csv": "4504bdb5210ed7bb2d0d5a1cf89373e527ee198cb54f9edd2a0e84782612c763",
        "sample labels": "f60df984b2ae3a31a10956808ff956145a59c70120162af3071dbd286398818b",
        "stain_channels.csv": "57eac57b213298e95a531be2b17336e17a65b956df0df92f0d0ec4750a7dee1a",
        "stain_objects.csv": "c6ec025e437d87e4ee4cc90c3841a94dfc0efd5b3e7f00be0c5ca5a53706ccaa",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "74e4a3df22e81ab74e519ab2870233a5839238e6727099253fe886437f2e0002",
    },
    "context-tissue2d-v1": {
        "bright_regions.csv": "d79714c3c13f45f310c2360fcc63e82f71c1508212ed6e23340d8dfc545c2f92",
        "cell_context.csv": "3bcebdb2b5c30fdc929da01700d6593ffa0d4c74196c5c1bdd6661703d5ce41a",
        "cell_fovs.csv": "9477d4692f26290fcaedbd794d23f52a1cf3e2b850184aa139dbf24bae0f9567",
        "cell_gene_counts.csv": "10609016ae6e09f257e2e72dbf8362c5c37c4f3042d83e7e1d24cb0926743464",
        "cells.csv": "aec5e5b14b3f04c60ea9090ec5bc1e76974fc5a7d8c4c57371fd47cbb1f36911",
        "context stains": "e0e474d7171fa887f0efb5bbee2e5e37957685c64f893a31028062b8ec26d560",
        "fov images": "bf5a5237a8639111a3d3d7fc7b8fa657cde139c36ef6ffb42cb009c18f27bf78",
        "fov labels": "3feedb067f1a9699212f13313cc339447f7a117d090befc7d3e4bb82f42d3245",
        "fov tables": "208ca148f61c847894b2c32b0fd4881d689ed9dc8ddb892431e1f9e06d389cfd",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "cc1f1b143e0a8b4aa3cf66f0238c271cb3ada4579198729d1ecfa704ce96ed0c",
        "marker_pairs.csv": "179daab6d78637862cb9c993a9c247bb3bce9658d2d6bb3e127d99f54d7b4a88",
        "molecule_fovs.csv": "72f14d029a4bd8b051d5335a83ffe3c6408d509cfd1f095918d8225957b7fdd5",
        "molecules.csv": "40d3f6cbc863a0fdcd213f1b6630bb465e74f4a8f518b33ceb5b8195c931af7f",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "63623bcd307d2f5499e3ca7252c1c37f81be9676ec39930f74b164ce69d294b9",
        "placement_truth.csv": "bb220e3e3b15f976264484c85a47960d2deb9138c8e7fe69a0e72fb11ef7c095",
        "positions.csv": "80dda74c0f64dee0bb8569a450bb315c2e83c4bc3ccaefe0d256c3a906566d08",
        "records": "76cbd55e17b96a731d1feacebb9ce31b458907a539ea4fe14b1742b4fdbfc346",
        "regions.csv": "2a19a47e877be2d29e0f748020b0d29912a936f672121b6310da51dc9068a578",
        "sample labels": "8b6f148d04fc376e3138f978354bc30f105033a8dbfb03f73684d6fba891fdbc",
        "stain_channels.csv": "b0edcde7497c38f742f17ddacfe4cf6aa498452517046f55ac2cc94f86850e5e",
        "stain_objects.csv": "8fda48015454786442fd12e4e23237aa6e3a0d48211e30cb50c2376f3234466e",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "6b09d8f00b86622eab59d0ee4671115a7d16d2eeb2e5f1dde97388d67920c296",
    },
    "context-culture-v1": {
        "bright_regions.csv": "b5a53b3884bf9bb55be460549734baf396a4de758237a1cf7ef0a38505aca94f",
        "cell_context.csv": "f2d56d4b242d1340b11e9145d56994df3afa7ff5240974a1ab647faec417901c",
        "cell_fovs.csv": "c5a13ca528dc318d554b538a155a6b5493a11ed9c029ac844c8c320a9dcd5de5",
        "cell_gene_counts.csv": "60c3e7bd2595b60c8551aed4869a7c1584c5d2fa837a558367c9e86d9e476d57",
        "cells.csv": "753a231472998d070088d2c787bb168e25c65d57f9e12d0f826aa20904240695",
        "context stains": "61a70c030f99d9b2bfe0662b5e5b114e2a981071a8fa6582913444ffac5a791b",
        "fov images": "e700c1986627abf5cf9a72b71e03318893e323179c747cf20dee401e4cf0a4bb",
        "fov labels": "1531b2a97bc18491a321c4ad29cebe61b072494e4215eb8a5e2f3b3ab505ecab",
        "fov tables": "abc63df93aea41179ba383bb153bb7ebfed396f3ee40a0fe4a816d49420fbf29",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "09378080133ef095855050fdf8430ec64e1125140d4d0d9481cf01df57e739be",
        "marker_pairs.csv": "b716e0aa20e63b7da7359e9dbc40131fc266e06fbdd944d45ca37b158923d358",
        "molecule_fovs.csv": "e2a715f7ac2927b8f49a5a96f332b0e7fb352aaf150b03ea02807e95d146cd36",
        "molecules.csv": "e811d792cb1893dd56fbf7762e2be1007ef9471f1636fa58a23e3c2cfa2b8eec",
        "morphology": "11dbd8c22af2d4755df46a7a71374cf1ca859d9804edf15c737b9ca09bbf6081",
        "morphology_truth.csv": "99861d0ce9b3070c8041346373713e4bd3e90c5eef1f8faa5c342329c4f2f6b6",
        "nuclei.csv": "9e7a467df3148a9e0fb68afb64aa1442e57415ae46f661798952c0aaec41b80e",
        "placement_truth.csv": "cce45aa67014d4f077b0c58e2b5bb06ad6ccf0fac383fe9952da5f72cd5bde5b",
        "positions.csv": "d290b28cf97125b29dd7218ecea2c6547ec3cd3e0076b55d56f9043a8f455783",
        "records": "4b97cdb1d4e12267f731281a5a252f8cb825a038dca0c1672c141ad73804a77a",
        "regions.csv": "ff008a75f16d6b341ffff7ba735e6b1fb60f6bac824b0516bed31cae429c08be",
        "sample labels": "85706dc7674c16229ce883be5c9c6c12bde31da9e91209aebd3d38e1621ae342",
        "stain_channels.csv": "5d81a7f67b7f61643dfb0adc5c0b2211e7b139baf5241c84dfa692212891d520",
        "stain_objects.csv": "5374312c7e5e226de5076aa586b50ed23f7744c2e620b3dc6021bef22fc1964d",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "a3c2fa272004d723fa4399bc60bac9d8194f9f815f632f375f49bd15563f9d78",
    },
    "context-dense3d-slow-v1": {
        "bright_regions.csv": "f0c0ea66fd019b11d43760ac6aff25fb34a95dd4378c025b69fc96ccd5e3979b",
        "cell_context.csv": "d63ee8054b0fb2eeff96abd0aa078b8283225ad4e5bc769fa1d27888bc7366a5",
        "cell_fovs.csv": "03c698fea72a15caedc320bd1c9cafe049f14da1fa16d2552b11cddd9bd9cc1f",
        "cell_gene_counts.csv": "62087b31609fbbc37214332feea3bfa617b63cd6c47e04450f51a49f90f5b3bf",
        "cells.csv": "70bfbce2295725f247a2ddcfedcfaaa04f267c0e6b7f30a032edd948c3fe9862",
        "context stains": "c665149ad9160b13e6bc7f530abf36e1ff705aaa4b30ab0c2a6fd7dd2177e3d1",
        "fov images": "29c5113bb4583be7b8a8dd4518330b11c5d3480f5342941c7dd5f7fdf898200c",
        "fov labels": "2ffe964c0fa3ab31cd6af0cecfa793a31a4a0c355f8199aeaf32da002d3cc6aa",
        "fov tables": "fd197f4672c6cc447a585565746bbf86ebd7339ff8f64d87018e22057fbad084",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "3791a797a1ca0ea93c5c67a075f765bd882233ec02901a43108c73a91d967c4b",
        "marker_pairs.csv": "b36070df1fe98b6cd482d20377bf504570224f0334bc2e2f042256456fd960b5",
        "molecule_fovs.csv": "b76cd89b02fe7cf012be4a7cb67abc4ce2007915af59f637c9f60afc55379552",
        "molecules.csv": "aeb32975c5d14ab67a905b7f7341b452596fcb6418824038170dfb3f3836458f",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "720cca3f5f4946b23fbc3836c608f8ed555974f2a639cefa5bf910fb8683305b",
        "placement_truth.csv": "a758f0928e62dd6e44eb423c1965125ff6cddca3ee23f33351518a37640cbbf7",
        "positions.csv": "e0f4955564543093588102d5f22eb595394626ec0fec17a7f42d59bda07e8d94",
        "records": "1d1b8711a390638e5b0d0e7313e7ad2e532202b03177de54a8556d0d9a498efc",
        "regions.csv": "bd1bfe254b56636ae99e74c1c2a46b727546314d3aea6da41ec3b0a021997724",
        "sample labels": "40352599da99ddb45e29a60114073291c65cde2602f565ed1e08392ff86e98de",
        "stain_channels.csv": "57eac57b213298e95a531be2b17336e17a65b956df0df92f0d0ec4750a7dee1a",
        "stain_objects.csv": "a6924d2d16db518efd479b15e990ed79135eff986d33e4046363e04563fbd672",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "74e4a3df22e81ab74e519ab2870233a5839238e6727099253fe886437f2e0002",
    },
    "context-tissue2d-slow-v1": {
        "bright_regions.csv": "e7e0ca1b8eb060759857b94576de7d567bc7cad19a313447174b41422365e328",
        "cell_context.csv": "c5e6dd88d2dab04221f7cc632c1bd1bc79510c918d7ad37b1eaf32fd38e8e3c2",
        "cell_fovs.csv": "a248b5690c9ffb17708ffe7a79cd955937ed101b57d8e6fa837011af7e7eca98",
        "cell_gene_counts.csv": "67a1025437a1dbbe41898c82ebb2afc7b0c7540fbec6fb2124a4b3489c3e911f",
        "cells.csv": "6219c57e44a967a681437c0bb35288dbd6c4791cdfa400d954fe453038a1ffb4",
        "context stains": "ffa2c1d83df12533710c0dc30db4e970a3896efce66b47f8ecb8fdc3755b2275",
        "fov images": "6e920e08ea867de08534eb0c0cb7b6d7e77dc80c304e993bd017aafeaa5b7d27",
        "fov labels": "ebdf0db65d548a8b8f03c6522d8b69c2dd0e5015b7366cad8ae53d6f254e7e63",
        "fov tables": "a66fdd4b9df52c019ef6ed8971692b8b86f750b2bdea807dd2d9f30627604db2",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "c71f607cccd6a616590e26369786a36505a1c8eee43144e43c7ab366a71c3201",
        "marker_pairs.csv": "7ca9789d3c95094b66243b46f3acfca645a133b43652333b529dbd3081a56716",
        "molecule_fovs.csv": "973779ac659b124fdeb21fe21f4055e4970f7196f17607c36c50620577afd31e",
        "molecules.csv": "c0cae3646cd341b45be41a5ebc8500f9e359fa8f375d150f39d6787ab01054fe",
        "morphology_truth.csv": "d01b3ae7e39d61eddd22d7ad2022777a0052f4ed20732ea4fd027dc88b1ec1bb",
        "nuclei.csv": "df6328720eefd49cf1edd602889d04cba71ac59721c02108cf394fee501d4bb5",
        "placement_truth.csv": "936d825c46cca5cf26f106612571a5281812747093c28f5d33a69455f3241352",
        "positions.csv": "6bc860148994d4b98a2a13c7fd12e12bc78b390967674f8fe781f14173d61f8b",
        "records": "f6371008a6511eb9e7362e32096d13241fdc7f41a652af9a31cbd3893581ea0b",
        "regions.csv": "55a9a9e3a8a107e7c2feef4457e78372893b9540c1b8748083be0b5dff89c8d8",
        "sample labels": "2e65a2c2cce5e7bb76ecc9f1dafa4af142b8d2e704adf8497c0bec3466b6a869",
        "stain_channels.csv": "b0edcde7497c38f742f17ddacfe4cf6aa498452517046f55ac2cc94f86850e5e",
        "stain_objects.csv": "669929a1629f66aac61ebf44f3e1ac4b001b99e67c4743c2059c2cb4c9329856",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "6cb2f6166041380a3cbf930cfbd5df8023a699b0064df7380d4a496e790a579c",
    },
    "context-culture-slow-v1": {
        "bright_regions.csv": "6b9ee01318353ca7d9b5b6b8fdaa67df1b6ffea1688176d81baea84a90f3712f",
        "cell_context.csv": "770864873edf4c086d181a1c99fc13112cb388ed95ccffc26e2773acae0ef3bd",
        "cell_fovs.csv": "b8f25bfb1d6946dc55f9e2306ee32f10584a1182bd1f3bf60bbd75c09e4eb4c6",
        "cell_gene_counts.csv": "eb547aef70fbad7c9a19548121c5b650603d2cc98273e23055a108af0945291c",
        "cells.csv": "089b06ee9981ac5ab9e4c481ba090ff5b9a1f2666ce42b73a5e292f85d2c07a3",
        "context stains": "4584e6cc1449444b711e99420f7edbedc3adf0a0f185d1a371e659ae9e52cd86",
        "fov images": "66be97a4644f6dd828ae9bb3b8e18ffff091b1ccd13ef37e75f279f4d8970adf",
        "fov labels": "ed21cdf525ec90d5b607f1945e9e39615b7f2978e27d2d71e897a03d0fc55f7f",
        "fov tables": "0ade037caa158fdc59c4d9a127d3c87a98fa39c24054bdd73d0b180e25af2b1c",
        "harmonics.csv": "f52c278693cd9a968520e97e37053d9a7f4e09d2cd02c2c2e29fdcf6f0f318a7",
        "lobes.csv": "ce5cfc14d7619dc146c95db9d813106a244c3dd5e1d80bfc757cc169fcd928cd",
        "marker_pairs.csv": "d34f81e795f807b75afd02a95135ca5b2c1ce5df57b826101aff24e01d10ff16",
        "molecule_fovs.csv": "5125abafd0c13b1a09761bc160aed146e29e4a5bfc2617ca2e7abe172e9efd0d",
        "molecules.csv": "01df17778121a5c0c68698e1f866b616bc0028cad166af77016247500a32ed57",
        "morphology": "d848e1899d72eb7f1b3fbb555e3bb98fe932d66afe191fa07be46ca8702e8449",
        "morphology_truth.csv": "33a1fd2a7a248d7abc56a57282f8608b22ceaef3cd3ca05c6b877e45761d07bf",
        "nuclei.csv": "2cfc83ee8b24bfebdebe26c83010afa876c3383f4c50ef358fb85cf6ce3452c3",
        "placement_truth.csv": "3e21910427d2407c866f4ed904c3980feecfd523041155313586eeaedf22dba5",
        "positions.csv": "5ab28104ba00c32ef20afb6b714858bcae5e73bd49810e8b76174dc9ae17f33c",
        "records": "9275d807281e5c85a53ec3c6b181824f229d0a68484c712d788f72a994d2d5cf",
        "regions.csv": "3537e2d504f6d0d82ead90f56340eea7650c44faf93fb9989a20e92028abbef5",
        "sample labels": "c7315110f0e9d058cd23b7ebf86d3f69ca49025dfd764f10cc8f0f7f8c5b9c36",
        "stain_channels.csv": "5d81a7f67b7f61643dfb0adc5c0b2211e7b139baf5241c84dfa692212891d520",
        "stain_objects.csv": "555b0267950ea376498ac868791479a0c2d309790e5c06ca7be8c360335c8702",
        "type_abundances.csv": "c8188156d49077abd44a922a76d91eb8b5346a4b4cd4dd1ba00ef858f92cc0e3",
        "unevenness.csv": "7cbee64cf0773204b036262971ffe1f50e488832889bb9aead918282f10d7fa0",
    },
}


def _number(value):
    """A float to 6 significant digits; magnitudes below 1e-9 are 0 (the last bit differs between CPUs)."""
    return 0.0 if abs(value) < 1e-9 else float(f"{value:.6g}")


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
    book, config = cellular_scene_preset(name)
    root = save_cellular_scene(generate_cellular_scene(book, config=config), tmp_path / name)
    digests = saved_digests(root)
    assert sorted(digests) == sorted(PINNED[name])
    changed = {group: digests[group] for group in PINNED[name] if digests[group] != PINNED[name][group]}
    assert not changed, f"{name}: groups whose saved content differs from the base-revision pins: {changed}"
