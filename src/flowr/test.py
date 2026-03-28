"""FlowR evaluation entry point. Usage: python -m flowr.test --config <path>"""
import runpy

runpy.run_module("flowr.training.evaluator", run_name="__main__")
