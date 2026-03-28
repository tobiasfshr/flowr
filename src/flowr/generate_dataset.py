"""Generate stage 2 dataset. Usage: python -m flowr.generate_dataset --model <path> --output_dir <dir>"""
import runpy

runpy.run_module("flowr.training.generate_dataset", run_name="__main__")
