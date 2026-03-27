import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images")

print("Path to dataset files:", path)
