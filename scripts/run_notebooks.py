"""
Script to execute all 14 notebooks in notebooks/ and save their outputs.
Uses nbformat and nbclient with the current Python environment.
"""

import os
import nbformat
from nbclient import NotebookClient

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
notebook_files = sorted([f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")])

print(f"Found {len(notebook_files)} notebooks in {notebook_dir}")

for nb_file in notebook_files:
    nb_path = os.path.join(notebook_dir, nb_file)
    print(f"Executing: {nb_file} ...", end=" ", flush=True)
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute(cwd=notebook_dir)

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print("SUCCESS")
    except Exception as e:
        print("FAILED:", str(e).encode("ascii", "replace").decode("ascii"))

