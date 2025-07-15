# Cyclone Intensity Estimator Using Satellite Imagery

## Overview

This project leverages deep learning to estimate the **intensity of tropical cyclones** using infrared satellite imagery from the **INSAT-3D** satellite. A web-based interface built with **Django** allows users to upload cyclone images and receive real-time predictions of cyclone intensity, measured in knots.

---

## Dataset

* **Source**: [INSAT3D Infrared Cyclone Dataset (2013–2021) – Kaggle](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)
* **Input Format**: Infrared `.png` images of cyclones over the Indian Ocean
* **Labels**: Cyclone intensities (in knots)

---

## Model Training (Jupyter Notebook)

All stages of model training—from preprocessing to evaluation—are implemented in a Jupyter Notebook.

### Training Pipeline

1. **Data Loading**

   * CSV containing image filenames and corresponding intensity labels

2. **Preprocessing**

   * Resized all images to 224×224
   * Normalized pixel values
   * Converted to PyTorch tensors

3. **Model Architecture**

   * Backbone: **EfficientNet-B0** pretrained on ImageNet
   * Final layer adapted for regression

4. **Training Configuration**

   * **Loss Function**: Mean Squared Error (MSE)
   * **Optimizer**: Adam
   * **Learning Rate Scheduler**: ReduceLROnPlateau
   * Includes Early Stopping and Model Checkpointing

5. **Evaluation Metrics**

   * Mean Absolute Error (MAE): \~9.2 knots
   * Root Mean Square Error (RMSE): \~10.7 knots
   * R² Score: \~0.84

| Metric | Value  |
| ------ | ------ |
| MAE    | 9.2    |
| RMSE   | 10.7   |
| R²     | \~0.84 |

---

## Web Application (Django)

An interactive Django web application allows users to upload infrared cyclone images and get predicted intensity values.

### Key Features

* Browser-based image upload
* Server-side preprocessing and inference using the trained model
* Clean, minimal frontend interface
* Real-time display of predicted cyclone intensity

### Workflow

1. User uploads an image through the frontend form
2. The image is stored and passed to a prediction script
3. The model loads `best_model.pt`, processes the image, and performs inference
4. Predicted intensity is returned and displayed to the user

---

## Running the Application Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Cyclone-Intensity.git
```

### Step 2: Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

### Step 3: Run the Django Development Server

```bash
cd Cyclone-Intensity
python manage.py runserver
```

Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the web interface.

---

## Requirements

* Python 3.10+
* PyTorch
* torchvision
* Pillow
* Django 5.x
* scikit-learn
* matplotlib, pandas, numpy

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Sample Output

![Cyclone-Image-1](https://github.com/user-attachments/assets/9b6f67d7-583c-48fb-817e-84fa17cb78da)

![Cyclone](https://github.com/user-attachments/assets/55a88c6e-75d9-4389-8799-dea8a872959b)

![Prediction](https://github.com/user-attachments/assets/1e77712d-d769-4359-b940-3e8044ad6183)

---

## Future Enhancements

### Data & Model Improvements

* Expand dataset with high-resolution, multi-basin satellite imagery (e.g., GOES, MODIS)
* Experiment with advanced architectures (EfficientNetV2, Swin Transformer)
* Implement regularization techniques and data augmentation
* Optimize hyperparameters using cross-validation

### Real-Time Integration

* Connect with satellite APIs (NOAA, NASA Earthdata, ISRO MOSDAC)
* Schedule automatic image ingestion using Celery or cron

### Feature Extensions

* Cyclone localization using object detection models (e.g., YOLOv8)
* Alert system (email/SMS/WhatsApp) for critical intensity thresholds
* Integration with mapping libraries (e.g., Mapbox, Leaflet.js) for visualization
* Combine with meteorological data using hybrid ML models (e.g., LSTM, GNN)

---

## Acknowledgements

* \[INSAT-3D Dataset – ISRO via IMD]
* PyTorch, EfficientNet, Django
* Kaggle Community

---
