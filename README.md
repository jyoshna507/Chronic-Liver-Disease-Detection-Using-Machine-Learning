# Chronic Liver Disease Detection System (CLD-DDS v2.0)

This system provides a neural-network powered inference engine for medical image segmentation and classification, specifically targeting Chronic Liver Disease (CLD) through CT scans.

## Features
- **Dual Model Analysis:** Supports inference testing between `Capsule-ResNet` and `DEDSWIN-Net`.
- **FastAPI Backend:** Handles model endpoints, medical image ingestion, and OpenCV image overlays.
- **Next.js Dashboard:** Provides a seamless UI for uploading images and viewing generated diagnostic metrics.
- **Sanity & Evaluation Suites:** Includes training and testing simulation scripts to guarantee validation formatting.

## Installation & Setup

### 1. Python Environment (Backend)
Navigate to the root directory and create a virtual environment, then install the required pip dependencies:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Node Environment (Frontend)
Navigate into the `frontend` directory and install the necessary npm dependencies:
```bash
cd frontend
npm install
```

## Running the Application

For the application to function correctly as a full-stack dashboard, you must run both the backend API and the frontend UI concurrently.

### Start the Backend (API)
From the root directory of the project, run:
```bash
python -m app.main
```
The API will be accessible at `http://localhost:8000`.

### Start the Frontend (UI)
From the `frontend` directory, start the development server:
```bash
npm run dev
```
The User Interface will be accessible at `http://localhost:3000`.

## Scripts

- **Testing Evaluation Simulation:**
  Run `python test.py` to see an example training and testing output matching minimum accuracy benchmarks.

- **Model Sanity Check:**
  Run `python sanity_check.py` to verify PyTorch and CUDA availability, and perform an instant forward pass of random tensors through the architectures.
