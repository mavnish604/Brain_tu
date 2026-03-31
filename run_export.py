import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from executorch.exir import to_edge

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "models").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root containing a models/ directory.")

PROJECT_ROOT = find_project_root(Path.cwd())
MODELS_DIR = PROJECT_ROOT / "models"
PTH_MODEL_PATH = MODELS_DIR / "densenet121_brain_tumor.pth"
PTE_MODEL_PATH = MODELS_DIR / "densenet121_brain_tumor.pte"

def build_model() -> torch.nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    state_dict = torch.load(PTH_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = build_model()
example_inputs = (torch.randn(1, 3, 224, 224),)

with torch.no_grad():
    eager_output = model(*example_inputs)

print("Exporting...")
exported_program = torch.export.export(model, example_inputs)
edge_program = to_edge(exported_program)
executorch_program = edge_program.to_executorch()

PTE_MODEL_PATH.write_bytes(executorch_program.buffer)
print(f"Saved ExecuTorch program to: {PTE_MODEL_PATH}")
