"""DL3DV data preparation. Usage: python -m flowr.prepare_dl3dv generate <work_dir> <data_dir> <scale_dir> ..."""
import runpy

runpy.run_module("flowr.scripts.datasets.dl3dv.generate_data", run_name="__main__")
