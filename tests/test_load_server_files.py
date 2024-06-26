import os
import pytest

from utils import try_load_file

__filepath_2013 = "/gpfs01/euler/data/Data/Franke/20130419/1/Raw/Q0_DN.smp"
__filepath_2013_2 = "/gpfs01/euler/data/Data/Franke/20130828/1/Raw/Q7_DN.smh"
__filepath_2014 = "/gpfs01/euler/data/Data/Franke/20140110/1/Raw/Q0_Chirps.smp"
__filepath_2015 = "/gpfs01/euler/data/Data/Franke/20150616/1/Raw/Q0_DN.smp"
__filepath_2016 = "/gpfs01/euler/data/Data/Franke/20160330/1/Raw/Q2_DarkBars.smp"
__filepath_2018 = "/gpfs01/euler/data/Data/Franke/20181015/1/Raw/M1_LR_GCL3_Chirp.smp"
__filepath_2020 = "/gpfs01/euler/data/Data/Gonschorek/20200609/1/Raw/M1_LR_GCL1_ChI_C2.smp"
__filepath_2022 = "/gpfs01/euler/data/Data/Gonschorek/20200416/1/Raw/M1_LR_GCL0_ChI_C1.smp"

__filepath_ipl_2018 = "/gpfs01/euler/data/Data/Szatko/20180720/2/Raw/M1_RR_IPL1_DN.smp"
__filepath_zstack_2018 = "/gpfs01/euler/data/Data/Franke/20180223/1/Raw/M1_P0_Dend_Stack.smp"


@pytest.mark.skipif(not os.path.isfile(__filepath_2013), reason="File not found")
def test_load_2013_file():
    try_load_file(__filepath_2013)


@pytest.mark.skipif(not os.path.isfile(__filepath_2013_2), reason="File not found")
def test_load_2013_2_file():
    try_load_file(__filepath_2013_2)


@pytest.mark.skipif(not os.path.isfile(__filepath_2014), reason="File not found")
def test_load_2014_file():
    try_load_file(__filepath_2014)


@pytest.mark.skipif(not os.path.isfile(__filepath_2015), reason="File not found")
def test_load_2015_file():
    try_load_file(__filepath_2015)


@pytest.mark.skipif(not os.path.isfile(__filepath_2016), reason="File not found")
def test_load_2016_file():
    try_load_file(__filepath_2016)


@pytest.mark.skipif(not os.path.isfile(__filepath_2018), reason="File not found")
def test_load_2018_file():
    try_load_file(__filepath_2018)


@pytest.mark.skipif(not os.path.isfile(__filepath_2020), reason="File not found")
def test_load_2020_file():
    try_load_file(__filepath_2020)


@pytest.mark.skipif(not os.path.isfile(__filepath_2022), reason="File not found")
def test_load_2022_file():
    try_load_file(__filepath_2022)


@pytest.mark.skipif(not os.path.isfile(__filepath_ipl_2018), reason="File not found")
def test_load_ipl_2018_file():
    try_load_file(__filepath_ipl_2018)


@pytest.mark.skipif(not os.path.isfile(__filepath_zstack_2018), reason="File not found")
def test_load_zstack_2018_file():
    try_load_file(__filepath_zstack_2018)
