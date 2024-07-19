# Singapore-Resale-Flat-Prices-Predicting
## Overview

  This project aims to construct a machine learning model and implement it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore. This prediction model will be based on past transactions involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety of criteria, including location, the kind of apartment, the total square footage, and the length of the lease. The provision of customers with an expected resale price based on these criteria is one of the ways in which a predictive model may assist in the overcoming of these obstacles.

## Features
 * About Project: Provides detailed information about the HDB resale market, including the resale process, valuation, eligibility criteria, and more.
 * Predictions: Allows users to input flat details and get a predicted resale price based on the trained machine learning model.

## Dependencies

 * Python 
   
 * Streamlit
   
 * streamlit_option_menu
   
 * pandas
   
 * geopy
   
 * statistics
   
 * numpy
   
 * scikit-learn
   
 * requests
   

## Usage

*  About Project
  
  Navigate to the "About Project" section in the sidebar to read about various aspects of the HDB resale market.
   
* Predictions
  
     *  Navigate to the "Predictions" section in the sidebar.
       
     *  Fill in the form with the following details
       
     *  Street Name: The street name of the flat.
       
     * Block Number: The block number of the flat.
       
     *  Floor Area (Per Square Meter): The floor area of the flat in square meters.
       
     *  Lease Commence Year: The year the lease commenced.
       
     * Storey Range: The range of storeys (e.g., "1 TO 5").
       
     *  Click on "PREDICT RESALE PRICE" to get the predicted resale price.
       
* Files
  
     * app.py: The main Streamlit application file.
   
     * mrt.csv: The CSV file containing MRT station coordinates.
   
     *  model.pkl: The trained machine learning model.
   
     * scaler.pkl: The scaler used for preprocessing input data.
 
## Model Training
The model is trained using a Decision Tree Regressor. Below is a summary of the steps involved in training the model:

   * Data Collection: The dataset is collected and preprocesseor
     
   * Feature Engineering: Features are extracted and engineered from the dataset.
     
   * Model Training: A Decision Tree Regressor is trained using the preprocessed data.
     
   * Hyperparameter Tuning: GridSearchCV is used for hyperparameter tuning to find the best model parameters.
     
   * Model Evaluation: The model is evaluated using Mean Squared Error (MSE).
     
   * Model Saving: The trained model and the scaler are saved as pickle files.
