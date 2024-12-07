# Car Price Prediction Project

## Overview

This project aims to predict the selling price of used cars based on various features such as brand, model, mileage, engine capacity, transmission type, fuel type, and more. The dataset used is from the "Car Dekho" platform, and two models — Random Forest Regressor and Linear Regression — are implemented to evaluate their performance in predicting car prices.

## Project Structure

- **Data Preprocessing**: The dataset is cleaned and processed by filling missing values, encoding categorical variables, and scaling the numerical features.
- **Model Building**: Random Forest and Linear Regression models are trained to predict car prices.
- **Evaluation**: The models are evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
- **Prediction**: The trained models can predict the price of a car based on input features.

## Features

- **brand**: Car brand
- **model**: Car model
- **vehicle_age**: Age of the car in years
- **km_driven**: Kilometers driven by the car
- **seller_type**: Type of seller (Individual/Dealer)
- **fuel_type**: Type of fuel used (Petrol/Diesel/CNG)
- **transmission_type**: Type of transmission (Manual/Automatic)
- **mileage**: Mileage of the car
- **engine**: Engine capacity
- **max_power**: Maximum power of the engine
- **seats**: Number of seats in the car
- **selling_price**: The target variable — selling price of the car

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `google-colab` (for uploading dataset in Google Colab)

You can install the required libraries by running the following:

```bash
pip install -q scikit-learn pandas numpy matplotlib seaborn
```

## Steps to Run

1. **Upload Dataset**: Upload the `cardekho_dataset.csv` file.
2. **Data Preprocessing**: Missing values are filled using the median of the respective column, and categorical variables are encoded using `LabelEncoder`. Numeric features are scaled using `StandardScaler`.
3. **Train/Test Split**: The dataset is split into training and testing sets (80%-20%).
4. **Model Training**: The models — Random Forest Regressor and Linear Regression — are trained on the training data.
5. **Model Evaluation**: Both models are evaluated using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score
6. **Car Price Prediction**: The model can predict the selling price based on input data for a car.

## Example

To predict the selling price of a car, you can input the following details:

```python
example_car = {
    'brand': 'Toyota',
    'model': 'Corolla',
    'vehicle_age': 5,
    'km_driven': 50000,
    'seller_type': 'Individual',
    'fuel_type': 'Petrol',
    'transmission_type': 'Manual',
    'mileage': 18,
    'engine': 1500,
    'max_power': 90,
    'seats': 5
}
```

This will give the predicted selling price based on the features provided.

## Results

The Random Forest Regressor and Linear Regression models are evaluated using the following metrics:

- **Random Forest Model**:
  - MAE, MSE, RMSE, R² Score
- **Linear Regression Model**:
  - MAE, MSE, RMSE, R² Score

You can compare the evaluation metrics to determine which model performs better.

## Conclusion

This project demonstrates how machine learning models can be used to predict car prices based on various features. The Random Forest Regressor model, being a non-linear model, typically performs better than the Linear Regression model. However, both models can provide valuable insights for predicting used car prices.

---

Feel free to modify the sections as per your project requirements.
