import os
import pytest

from scanmsupport.scanm.scanm_smp import SMP

filepath_2013 = "/gpfs01/euler/data/Data/Franke/20130419/1/Raw/Q0_DN.smp"
filepath_2022 = "/gpfs01/euler/data/Data/Gonschorek/20200416/1/Raw/M1_LR_GCL0_ChI_C1.smp"


@pytest.mark.skipif(not os.path.isfile(filepath_2013), reason="File not found")
def test_load_2013_file():
    try:
        scmf = SMP()
        scmf.loadSMH(filepath_2013, verbose=False)
        scmf.loadSMP(filepath_2013)
    except Exception as e:
        assert False, f"Reading the file raised the error\n{e}"


@pytest.mark.skipif(not os.path.isfile(filepath_2022), reason="File not found")
def test_load_2020_file():
    try:
        scmf = SMP()
        scmf.loadSMH(filepath_2022, verbose=False)
        scmf.loadSMP(filepath_2022)
    except Exception as e:
        assert False, f"Reading the file raised the error\n{e}"
