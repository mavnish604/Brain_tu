import json

with open("Notebooks/pth_to_pte.ipynb", "r") as f:
    nb = json.load(f)

# Keep the markdown cell (0), the basic imports (1), the model loader (3 -> now 2), and the export logic (4 -> now 3)
new_cells = []

# cell 0: Markdown
new_cells.append(nb["cells"][0])

# cell 1: Dependency check & basic imports
cell1 = {
 "cell_type": "code",
 "execution_count": None,
 "id": "imports_and_checks",
 "metadata": {},
 "outputs": [],
 "source": [
  "import os\n",
  "# To avoid some CUDA symbol issues during export if present\n",
  "os.environ['CUDA_MODULE_LOADING'] = 'LAZY'\n",
  "\n",
  "from pathlib import Path\n",
  "\n",
  "try:\n",
  "    import torch\n",
  "    import torch.nn as nn\n",
  "    from torchvision import models\n",
  "    from executorch.exir import to_edge\n",
  "except ImportError as exc:\n",
  "    raise ImportError(\n",
  "        \"Missing dependencies. Please ensure `torch`, `torchvision`, and `executorch` are installed.\"\n",
  "    ) from exc\n",
  "\n",
  "def find_project_root(start: Path) -> Path:\n",
  "    for candidate in [start, *start.parents]:\n",
  "        if (candidate / \"models\").exists():\n",
  "            return candidate\n",
  "    raise FileNotFoundError(\"Could not find the project root containing a models/ directory.\")\n",
  "\n",
  "PROJECT_ROOT = find_project_root(Path.cwd())\n",
  "MODELS_DIR = PROJECT_ROOT / \"models\"\n",
  "PTH_MODEL_PATH = MODELS_DIR / \"densenet121_brain_tumor.pth\"\n",
  "PTE_MODEL_PATH = MODELS_DIR / \"densenet121_brain_tumor.pte\"\n",
  "\n",
  "print(\"Project root:\", PROJECT_ROOT)\n",
  "print(\"Input checkpoint:\", PTH_MODEL_PATH)\n",
  "print(\"Output program:\", PTE_MODEL_PATH)\n"
 ]
}
new_cells.append(cell1)

# cell 2: Model loader (was cell 3)
cell2 = nb["cells"][3]
new_cells.append(cell2)

# cell 3: Export logic (was cell 4)
cell3 = nb["cells"][4]
# Remove the try/except block for executorch since we moved it to cell 1
cell3["source"] = [
 "exported_program = torch.export.export(model, example_inputs)\n",
 "edge_program = to_edge(exported_program)\n",
 "executorch_program = edge_program.to_executorch()\n",
 "\n",
 "PTE_MODEL_PATH.write_bytes(executorch_program.buffer)\n",
 "\n",
 "print(f\"Saved ExecuTorch program to: {PTE_MODEL_PATH}\")\n",
 "print(f\"File size: {PTE_MODEL_PATH.stat().st_size / 1e6:.2f} MB\")\n"
]
new_cells.append(cell3)

nb["cells"] = new_cells

with open("Notebooks/pth_to_pte.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
