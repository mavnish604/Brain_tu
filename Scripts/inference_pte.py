import argparse
import time
from pathlib import Path

# Add lazy init to avoid CUDA load issues on edge
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    from executorch.extension.pybindings.portable_lib import _load_for_executorch as Module
except ImportError as e:
    raise ImportError(
        "Required packages are missing. Make sure torch, torchvision, and executorch are installed in your environment."
    ) from e

class BrainTumorClassifierPTE:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        print(f"Loading ExecuTorch model from {self.model_path}...")
        start_time = time.time()
        # Load the compiled .pte model
        self.module = Module(str(self.model_path))
        print(f"Model loaded in {time.time() - start_time:.3f} seconds.")
        
        # DenseNet121 standard transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization typically used for torchvision models (ImageNet mean/std)
            # If your custom training script used different normalization, update this.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Based on the typical class structure for this dataset
        # DenseNet121 outputs 4 classes (as seen in pth_to_pte.ipynb: in_features, 4)
        self.classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    def predict(self, image_path: str):
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        print(f"Processing image: {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms and add batch dimension (1, C, H, W)
        input_tensor = self.transform(image).unsqueeze(0)
        
        print("Running inference...")
        start_time = time.time()
        
        # Run the model forward pass. 
        # Note: ExecuTorch module.forward() accepts a tuple of inputs and returns a tuple.
        output = self.module.forward((input_tensor,))
        raw_logits = output[0]
        
        inference_time = time.time() - start_time
        print(f"Inference complete in {inference_time:.4f} seconds.")
        
        # Calculate probabilities
        probabilities = F.softmax(raw_logits, dim=1)[0]
        
        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = self.classes[predicted_idx.item()]
        
        print("\n--- Results ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence:      {confidence.item() * 100:.2f}%")
        print("\nAll Probabilities:")
        for i, class_name in enumerate(self.classes):
            print(f"  {class_name:<10}: {probabilities[i].item() * 100:.2f}%")
            
        return predicted_class, confidence.item()

def main():
    parser = argparse.ArgumentParser(description="ExecuTorch (.pte) Inference Script for DenseNet121 Brain Tumor Model")
    parser.add_argument("--model", type=str, default="models/densenet121_brain_tumor.pte", help="Path to the .pte model file")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    
    args = parser.parse_args()
    
    try:
        classifier = BrainTumorClassifierPTE(args.model)
        classifier.predict(args.image)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
