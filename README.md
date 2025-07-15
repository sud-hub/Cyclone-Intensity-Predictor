## 🌪️ Cyclone Intensity Estimator using Satellite Imagery

### 🚀 Overview

This project uses satellite infrared imagery from the **INSAT-3D** satellite to estimate the **intensity of tropical cyclones** (in knots) using **deep learning**. A web-based Django interface allows users to upload cyclone images and get real-time intensity predictions powered by the trained model.

---

## 📊 Dataset

* **Source**: [Kaggle - INSAT3D Infrared Cyclone Dataset (2013–2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)
* **Input**: Infrared calibrated cyclone images (`.png`)
* **Label**: Cyclone Intensity (in knots)

> The dataset includes satellite imagery of cyclones over the Indian Ocean, labeled with their intensities.

---

## 🧠 Model Training (Jupyter Notebook)

The notebook handles everything from data preprocessing to model evaluation.

### 🔧 Steps

1. **Load CSV**: Contains image filenames and intensity labels.
2. **Preprocessing**:

   * Resized all images to 224×224
   * Normalized pixel values
   * Converted to PyTorch tensors
3. **Model**: Used **EfficientNet-B0** (pretrained on ImageNet), with final layer adapted for regression.
4. **Training Details**:

   * Loss: MSELoss
   * Optimizer: Adam
   * Scheduler: ReduceLROnPlateau
   * Early Stopping and Model Checkpointing added
5. **Evaluation**:

   * Final MAE: **\~9.2 knots**
   * Final RMSE: **\~10.7 knots**

### 🧪 Metrics

| Metric | Value  |
| ------ | ------ |
| MAE    | 9.2    |
| RMSE   | 10.7   |
| R²     | \~0.84 |

---

## 🌐 Django Web Application

A user-friendly frontend allows users to **upload a satellite image**, and the app returns the predicted cyclone intensity.

### 🔨 Features

* Image upload via browser
* Server-side prediction using trained model
* Clean and minimal HTML form
* Result rendered with cyclone intensity

### 🔄 Flow

1. User uploads image via form
2. Image is saved and passed to `predict.py`
3. `predict.py` loads `best_model.pt` and preprocesses image
4. Model predicts intensity and sends it back to frontend

---

## 💻 Running the App Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Cyclone-Intensity.git
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Run Django server

```bash
cd Cyclone-Intensity
python manage.py runserver
```

Go to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🧠 Requirements

* Python 3.10+
* PyTorch
* torchvision
* Pillow
* Django 5.x
* scikit-learn
* matplotlib, pandas, numpy

You can install everything using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

* **Larger Dataset**: Train on extended and higher-resolution satellite datasets (e.g. from GOES, MODIS, ISRO) across multiple ocean basins for better generalization.

* **Model Enhancements**:

  * Add regularization (dropout, label smoothing, weight decay)
  * Try stronger architectures (EfficientNetV2, Swin Transformer)
  * Use data augmentation and hyperparameter tuning

* **Automated Image Ingestion**:

  * Integrate satellite APIs (NOAA, NASA Earthdata, ISRO MOSDAC)
  * Schedule real-time data fetching using Celery or cron jobs

* **Cyclone Detection**:

  * Use object detection (YOLOv8) to localize cyclone regions from raw images
  * Optionally use segmentation for more precise structures

* **Real-time Alert System**:

  * Trigger alerts (email/SMS/WhatsApp) when intensity crosses critical thresholds
  * Add region tagging and basic geolocation tracking

* **Interactive Visualization**:

  * Overlay predictions on a map (Leaflet.js, Mapbox)
  * Display intensity heatmaps and historical trends

* **Hybrid Modeling**:

  * Fuse ML with meteorological features (wind, SST, pressure)
  * Explore GNNs and LSTMs for spatio-temporal predictions

---

## 📸 Sample Prediction Output

```text
Uploaded Image: cyclone_2019_10_28.png
Predicted Intensity: 102.4 knots
```

---

## 👨‍💻 Author

**Sudarshan Khot**
*Student, Machine Learning & Web Enthusiast*

---

## 🧠 Acknowledgements

* \[INSAT-3D Dataset - ISRO via IMD]
* PyTorch, EfficientNet, Django
* Kaggle Community

---
