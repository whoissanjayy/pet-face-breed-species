# Pet Face Breed & Species Classifier

An internship-ready Deep Learning project that classifies an uploaded cat/dog image into one of the 37 breeds in the Oxford-IIIT Pet Dataset.

## Features
- Streamlit web interface
- Cat/Dog species detection from the predicted breed
- 37-breed classification
- Top-K predictions
- Confidence score
- Image preview
- Optional image center-cropping
- ResNet18 transfer learning with PyTorch
- Automatic dataset download through torchvision
- Best-model checkpoint saving

## Project structure

```text
pet-face-breed-species-classifier/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── models/
│   └── (best_model.pth will be created after training)
└── utils/
    ├── __init__.py
    └── labels.py
```

## 1. Install Python

Use Python 3.10 or 3.11 on Windows.

Check:

```powershell
python --version
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Train the model

This downloads the Oxford-IIIT Pet Dataset automatically.

```powershell
python train.py
```

For a first run, use the default settings. The script creates:

```text
models/best_model.pth
models/class_names.json
models/training_history.json
```

Training time depends heavily on your CPU/GPU.

### Faster testing

For a quick pipeline test before a full training run:

```powershell
python train.py --epochs 1 --batch-size 16
```

This is only a smoke test, not a final-quality model.

## 5. Test prediction from terminal

After training:

```powershell
python predict.py path\to\your\cat_or_dog.jpg
```

Example:

```powershell
python predict.py sample.jpg
```

## 6. Run the web application

```powershell
streamlit run app.py
```

The browser should open a local address similar to:

```text
http://localhost:8501
```

## 7. GitHub

After confirming the application works:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit - Pet Face Breed and Species Classifier"
git remote add origin https://github.com/YOUR_USERNAME/pet-face-breed-species-classifier.git
git push -u origin main
```

Do not upload secrets, passwords, or API keys.

## Important note about model accuracy

A wrong prediction does not automatically mean the application code is broken. Breed classification quality depends on training time, image quality, preprocessing, class balance, and hardware. Always report the actual validation/test accuracy obtained by your training run rather than inventing a number.

## Internship presentation flow

1. Problem statement
2. Dataset: Oxford-IIIT Pet Dataset
3. Data preprocessing
4. ResNet18 transfer learning
5. Training and validation
6. Model evaluation
7. Streamlit deployment
8. Live prediction demo
9. Limitations and future improvements
