# src/simicpipeline/_jupyter.py
import os
import subprocess

def jupyter():
    cmd = [
        "jupyter", "notebook",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=",
        "--NotebookApp.password=",
    ]
    subprocess.run(cmd, check=True)
