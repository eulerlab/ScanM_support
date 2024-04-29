import numpy as np
import pytest

from utils import try_load_file, load_h5_file

import os
test_file_dir = os.path.dirname(os.path.abspath(__file__))

__filepath_xy = os.path.join(test_file_dir, "..", "data", "xy_scan", "M1_LR_GCL4_chirp.smp")
__filepath_xy_h5 = os.path.join(os.path.dirname(__filepath_xy), 'SMP_' + os.path.basename(__filepath_xy).replace('.smp', '.h5'))
__filepath_xz = os.path.join(test_file_dir, "..", "data", "xz_scan", "M1_LR_xz0_BCnoise_C1.smp")
__filepath_xz_h5 = os.path.join(os.path.dirname(__filepath_xz), 'SMP_' + os.path.basename(__filepath_xz).replace('.smp', '.h5'))


@pytest.mark.skipif(not os.path.isfile(__filepath_xy), reason="File not found")
def test_load_xy_file():
    try_load_file(__filepath_xy)


@pytest.mark.skipif(not os.path.isfile(__filepath_xz), reason="File not found")
def test_load_xz_file():
    try_load_file(__filepath_xz)


@pytest.mark.skipif(not (os.path.isfile(__filepath_xy_h5) or os.path.isfile(__filepath_xy)), reason="File not found")
def test_data_xy_file_compared_to_h5():
    scmf = try_load_file(__filepath_xy)

    h5_data = load_h5_file(__filepath_xy_h5)

    wDataCh0 = scmf.getData(0, crop=True).T
    wDataCh1 = scmf.getData(1, crop=True).T
    wDataCh2 = scmf.getData(2, crop=True).T

    assert wDataCh0.shape == h5_data['wDataCh0'].shape
    assert wDataCh1.shape == h5_data['wDataCh1'].shape
    assert wDataCh2.shape == h5_data['wDataCh2'].shape

    assert np.allclose(wDataCh0, h5_data['wDataCh0'])
    assert np.allclose(wDataCh1, h5_data['wDataCh1'])
    assert np.allclose(wDataCh2, h5_data['wDataCh2'])


@pytest.mark.skipif(not (os.path.isfile(__filepath_xz_h5) or os.path.isfile(__filepath_xz)), reason="File not found")
def test_data_xz_file_compared_to_h5():
    scmf = try_load_file(__filepath_xz)
    h5_data = load_h5_file(__filepath_xz_h5)

    wDataCh0 = scmf.getData(0, crop=True).T
    wDataCh1 = scmf.getData(1, crop=True).T
    wDataCh2 = scmf.getData(2, crop=True).T

    assert wDataCh0.shape == h5_data['wDataCh0'].shape
    assert wDataCh1.shape == h5_data['wDataCh1'].shape
    assert wDataCh2.shape == h5_data['wDataCh2'].shape

    assert np.allclose(wDataCh0, h5_data['wDataCh0'])
    assert np.allclose(wDataCh1, h5_data['wDataCh1'])
    assert np.allclose(wDataCh2, h5_data['wDataCh2'])
