#@ File(label="Saved delivery directory", style="directory") delivery
#@ Boolean(label="Show images", value=false) show_images
"""Run in Fiji's Script Editor (Python) or ImageJ --headless --run.

Uses installed HDF5_Vibez and ImageJ TIFF readers; never regenerates inputs.
The plugin's assumed micrometre calibration is explicitly removed because the
fixture has no physical calibration. Slice labels carry round/channel/zero-based Z.
"""
from ij import IJ
from ij.measure import Calibration
from sc.fiji.hdf5 import HDF5ImageJ
from java.lang import System
import json
import math
import os

root = str(delivery)
output = os.path.join(root, 'fiji-verification.json')
if not show_images:
    assert not os.path.exists(output), 'Preserve viewer evidence; choose a new delivery'
results = []
rounds = ['round10', 'round2', 'round1']
channels = ['ch02', 'ch00', 'ch03', 'ch01']
for depth in [9, 1]:
    case = os.path.join(root, 'z%d' % depth)
    for r, name in enumerate(rounds):
        dataset = '/layers/layer%04d/image' % r
        hdf = HDF5ImageJ.hdf5read(os.path.join(case, 'registered', 'images.h5'), dataset, 'zyxc')
        tiff = IJ.openImage(os.path.join(case, 'inspection', name + '.tif'))
        assert hdf is not None and tiff is not None
        for kind, imp in [('HDF5', hdf), ('TIFF', tiff)]:
            assert list(imp.getDimensions()) == [32, 32, 4, depth, 1]
            assert imp.getBitDepth() == 32
            imp.setTitle('z%d | %s | %s' % (depth, name, kind))
            # HDF5_Vibez assumes 1 um without element_size_um. Never retain this fiction.
            calibration = Calibration()
            calibration.setUnit('pixel')
            calibration.pixelWidth = calibration.pixelHeight = calibration.pixelDepth = 1.0
            imp.setCalibration(calibration)
            maximum_error = 0.0
            for z in range(depth):
                for c in range(4):
                    index = imp.getStackIndex(c+1, z+1, 1)
                    label = '%s | %s | z=%d' % (name, channels[c], z)
                    if kind == 'TIFF':
                        assert imp.getStack().getSliceLabel(index) == label
                    else:
                        imp.getStack().setSliceLabel(label, index)
                    processor = imp.getStack().getProcessor(index)
                    other = tiff.getStack().getProcessor(index)
                    for y in range(32):
                        for x in range(32):
                            value = processor.getf(x,y)
                            assert value == other.getf(x,y), 'HDF5/TIFF voxel mismatch'
                            expected = 0.0
                            for center, mapping in [(10, [1,0,3]), (22, [0,1,2])]:
                                if c == mapping[r]:
                                    radius = (z-depth//2)**2 + ((y-center)/1.25)**2 + ((x-center)/1.25)**2
                                    if radius <= 16:
                                        expected += 8*math.exp(-radius/2)
                            maximum_error = max(maximum_error, abs(value-expected))
            assert maximum_error < 1e-6
            imp.setPosition([2,1,4][r], depth//2+1, 1)
            imp.setDisplayRange(0,8)
            if not show_images:
                IJ.saveAs(imp, 'PNG', os.path.join(case, 'inspection', name+'-'+kind+'-fiji.png'))
            results.append(dict(case='z%d' % depth, round=name, format=kind,
                dimensions_xyczt=list(imp.getDimensions()), bit_depth=imp.getBitDepth(),
                max_absolute_error=maximum_error, all_voxels_hdf5_tiff_equal=True,
                channel_labels=channels, calibration='unknown; unit pixel; plugin default removed',
                hdf5_dataset=dataset, layout='zyxc', labels='round | channel | zero-based z'))
            if show_images:
                imp.show()
            else:
                imp.close()
if not show_images:
    with open(output, 'w') as stream:
        json.dump(dict(imagej=IJ.getVersion(), java=System.getProperty('java.version'),
        reader='HDF5_Vibez custom zyxc and ImageJ TIFF',
        mode='GUI' if show_images else 'Fiji headless application',
        results=results, status='passed'), stream, indent=2)
print('FIJI_VERIFICATION_PASS '+output)
if not show_images:
    System.exit(0)
