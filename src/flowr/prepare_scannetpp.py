"""ScanNet++ data preparation. Usage: python -m flowr.prepare_scannetpp generate <root> <work_dir> <data_dir> ..."""
import runpy

runpy.run_module("flowr.scripts.datasets.scannetpp.generate_data", run_name="__main__")
