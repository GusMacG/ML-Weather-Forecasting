# Weather Forecasting Using Machine Learning

## Overview
This repository contains the codebase employed for our research on accurate weather forecasting in the city of Monterrey, Mexico, as detailed in our paper titled "Harnessing Machine Learning for Reliable Weather Forecasting: Meteorological Impact on Sustainable Energy in Monterrey". The study evaluates the accuracy of various machine learning models including Random Forest, Support Vector Machines, Gradient Boosting, and LSTM neural networks in the prediction of key weather features for small geographical areas in 24hr time steps, with varying forecasting ranges.

## Key Features
- **Model Implementation**: Implementation of Random Forest, Support Vector Regression, Gradient Boosting, and LSTM models.
- **Hyperparameter Tuning**: Use of Keras Tuner to optimize LSTM parameters.
- **Data Preprocessing**: Scripts for cleaning and preprocessing the dataset.
- **Performance Evaluation**: Evaluation of model performance using metrics like MAE, MSE, and R2.
- **Forecasting**: Tools for forecasting weather conditions several days in advance using iterative prediction models.

## Installation
To set up the project environment:
```bash
git clone https://github.com/GusMacG/ML-Weather-Forecasting
cd \destination-folder
pip install -r requirements.txt

## Running the Notebooks
- **LSTM Tuner Hypermodel.ipynb**: Demonstrates the hyperparameter tuning process for the LSTM model.
- **Modifications_to_Monterrey_ML_Weather_Prediction.ipynb**: Outlines the methodology for trainning traidtional machine learning models and the computation of results.

## Acknowledgments

This project was supported by the Department of Sciences and Engineering at the Instituto Tecnológico y de Estudios Superiores de Monterrey. Special thanks to the Observatorio Cuenca De Río Bravo and the meteorologists at Sistema Meteorológico Nacional, for providing the dataset used in this study.