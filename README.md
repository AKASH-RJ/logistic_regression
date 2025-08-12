# Classification using Logistic Regression & Flask

## Overview

This project predicts **binary classification outcomes** using the **Logistic Regression** algorithm. The model is trained on a labeled dataset and deployed as a **Flask web application** with HTML & CSS.

-----

## Features

  - **Logistic Regression classifier** for binary classification problems.
  - **Flask backend** to serve predictions.
  - **Interactive HTML form** for user input.
  - **CSS styling** for a professional interface.
  - **CSV dataset** with labeled data for training.

-----

## Project Structure

```
logistic_regression_app/
│
├── model.py             # Trains and saves the Logistic Regression model
├── app.py               # Flask application for predictions
├── templates/
│   ├── index.html       # Main input form
│   └── result.html      # Displays prediction
├── static/
│   └── style.css        # CSS for styling
├── dataset.csv          # Dataset (training data)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains labeled examples for binary classification.

Example:

```
feature1,feature2,feature3,label
2.5,1.2,3.1,0
1.8,0.5,2.8,1
```

Features:

  - `feature1`, `feature2`, `feature3`: Input variables.
  - `label`: `0` or `1` (binary target).

-----

## How It Works

### Model Training (`model.py`)

  - Loads dataset from `dataset.csv`.
  - Splits data into training and testing sets.
  - Trains a Logistic Regression classifier.
  - Saves trained model as `model.pkl`.

### Web Application (`app.py`)

  - Loads `model.pkl`.
  - Accepts input from an HTML form.
  - Predicts the output class (`0` or `1`).
  - Displays the prediction result.

-----

## Running the Project

1.  **Train the Model**
    ```bash
    python model.py
    ```
2.  **Run Flask App**
    ```bash
    python app.py
    ```
3.  **Open in Browser**
    Go to: `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page

<img width="598" height="445" alt="Screenshot 2025-08-12 120819" src="https://github.com/user-attachments/assets/8b958f11-5465-4817-ae9d-a16cacb41cea" />

---
Prediction Result

<img width="611" height="521" alt="Screenshot 2025-08-12 120827" src="https://github.com/user-attachments/assets/38205a2d-32dd-4328-8bce-d8a94f2f0c08" />
