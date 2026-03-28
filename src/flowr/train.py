"""FlowR training entry point. Usage: python -m flowr.train --config-path=configs --config-name=flowr-512.yaml"""
import runpy

runpy.run_module("flowr.training.trainer", run_name="__main__")
