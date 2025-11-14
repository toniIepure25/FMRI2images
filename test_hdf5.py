#!/usr/bin/env python3
import h5py

try:
    f = h5py.File('cache/nsd_hdf5/nsd_stimuli.hdf5', 'r')
    print('Keys:', list(f.keys()))
    print('imgBrick shape:', f['imgBrick'].shape)
    f.close()
    print('✅ File is valid and accessible!')
except Exception as e:
    print(f'❌ File is corrupted: {e}')
