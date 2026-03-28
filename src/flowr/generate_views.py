"""Generate novel views with FlowR. Usage: python -m flowr.generate_views --config <path> --input_dir <dir>"""
import runpy

runpy.run_module("flowr.training.generate_views", run_name="__main__")
