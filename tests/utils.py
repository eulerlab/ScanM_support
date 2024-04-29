from scanmsupport.scanm.scanm_smp import SMP


def load_file(filepath):
    scmf = SMP()
    err_load_smh = scmf.loadSMH(filepath, verbose=False)
    err_load_smp = scmf.loadSMP(filepath)
    return scmf, err_load_smh, err_load_smp


def try_load_file(filepath):
    try:
        scmf, err_load_smh, err_load_smp = load_file(filepath)
        assert isinstance(scmf, SMP), "File not loaded"
        assert err_load_smh == 0, "SMH not loaded"
        assert err_load_smp == 0, "SMP not loaded"
    except Exception as e:
        assert False, f"Reading the file raised the error\n{e}"
    return scmf


def load_h5_file(filepath):
    import h5py
    with h5py.File(filepath, "r") as h5f:
        keys = list(h5f.keys())
        data_dict = {key: h5f[key][()] for key in keys}
    return data_dict