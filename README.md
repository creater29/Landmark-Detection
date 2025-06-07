# Landmark-Detection
This repository contains code for downloading the Google Landmark Dataset V2 Micro using KaggleHub, performing data analysis, and training a VGG19 Keras model for image classification.

#Colab Link: https://colab.research.google.com/drive/1mfvGrkRRbWO0Drk3TXbqyDmrn3_GITFe?usp=sharing

## Features
- Downloads dataset from Kaggle with `kagglehub`
- Data exploration and visualization with pandas and matplotlib
- Keras VGG19 model setup and training (from scratch, no pre-trained weights)
- Data batching and label encoding with error handling

## Requirements

- Python 3.7+
- KaggleHub
- NumPy
- pandas
- Keras (with TensorFlow backend)
- OpenCV
- matplotlib
- scikit-learn
- Pillow

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the notebook (`main.ipynb`) and open it in Colab or Jupyter.
2. Run all cells to download the dataset, analyze, and train the model.

## Notes

- Make sure your Kaggle API credentials are set up if running locally.
- The dataset is downloaded to `/content` by default (adjust paths as needed for your environment).
