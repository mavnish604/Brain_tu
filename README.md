# Brain Tumor MRI Classification

This repository contains a brain tumor MRI classification project built around a DenseNet121 model. It includes:

- training notebooks
- a local inference script
- a FastAPI backend
- a Next.js frontend for MRI image upload and prediction

The backend compares two models on the same MRI image:

- the original PyTorch checkpoint: `models/densenet121_brain_tumor.pth`
- the mobile-compatible TorchScript Lite model: `models/brain_tumor_final_universal.ptl`

## Repository Structure

```text
Brain_tu/
├── Dataset/
├── Notebooks/
├── Scripts/
│   ├── api.py
│   ├── download.py
│   ├── infernce.py
│   └── requirements-api.txt
├── frontend/
├── models/
├── venv/
├── LICENSE
└── README.md
```

## Requirements

- Python 3.12
- Node.js 20+
- npm

The project already uses a local virtual environment in `venv/`.

## Python Environment

From the repository root:

```bash
cd /media/tst_imperial/Projects/Brain_tumor/Brain_tu
source venv/bin/activate
```

Check that the virtual environment is active:

```bash
which python
which pip
python -c "import torch; print(torch.__version__)"
```

## Install Backend Dependencies

```bash
cd /media/tst_imperial/Projects/Brain_tumor/Brain_tu
source venv/bin/activate
python -m pip install -r Scripts/requirements-api.txt
```

## Run the Backend

Start the FastAPI server from the repository root:

```bash
cd /media/tst_imperial/Projects/Brain_tumor/Brain_tu
source venv/bin/activate
fastapi dev Scripts/api.py --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

For a production-style run:

```bash
fastapi run Scripts/api.py --host 0.0.0.0 --port 8000
```

## Run the Frontend

The frontend lives in `frontend/` and talks to the FastAPI backend.

```bash
cd /media/tst_imperial/Projects/Brain_tumor/Brain_tu/frontend
cp .env.example .env.local
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The default frontend API target is:

```text
http://127.0.0.1:8000
```

You can change it in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Run Local Inference

To run the local comparison script on the default MRI image:

```bash
cd /media/tst_imperial/Projects/Brain_tumor/Brain_tu
source venv/bin/activate
python Scripts/infernce.py
```

To test another image:

```bash
python Scripts/infernce.py --image "/absolute/path/to/image.jpg"
```

## API Prediction Example

Send an image directly to the backend with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/media/tst_imperial/Projects/Brain_tumor/Brain_tu/Dataset/Challenging Datasets/Challenging Datasets/Blurred Dataset/Glioma/bilateral_glioma (1).jpg"
```

## Dataset

The dataset is expected under:

```text
Dataset/
```

The repository includes a helper script:

```bash
python Scripts/download.py
```

If you download or replace the dataset manually, keep the existing directory layout expected by the notebooks and scripts.

## Notes

- The inference script filename is currently `infernce.py` and is kept as-is to match the existing project structure.
- The frontend requires the backend to be running before image upload will work.
- The backend loads both the original model and the mobile model during startup.

## License

See [LICENSE](LICENSE).
