import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained models and scalers
with open('churn_model.pkl', 'rb') as model_file:
    churn_model = pickle.load(model_file)

with open('scaler_churn.pkl', 'rb') as scaler_file:
    churn_scaler = pickle.load(scaler_file)

# Load the pre-trained Loan model and scaler
rf_model = joblib.load('loan_model.pkl')
loan_scaler = joblib.load('scaler_rf.pkl')

# Load clustering models and scalers from .pkl files
with open('scaler_cluster.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

with open('pca_cluster.pkl', 'rb') as pca_file:
    final_pc_input = pickle.load(pca_file)

with open('kmeans_cluster.pkl', 'rb') as kmeans_file:
    km_4 = pickle.load(kmeans_file)

# Load the dummies.pkl file to match dummy columns in the input data
with open('dummies.pkl', 'rb') as dummies_file:
    dummies = pickle.load(dummies_file)

# Purchasetype function for clustering
def purchasetype(x):    
    if (x["ONEOFF_PURCHASES"] == 0) & (x["INSTALLMENTS_PURCHASES"] == 0):
        return("NONE")
    elif (x["ONEOFF_PURCHASES"] > 0) & (x["INSTALLMENTS_PURCHASES"] == 0):
        return("ONEOFF")
    elif (x["ONEOFF_PURCHASES"] == 0) & (x["INSTALLMENTS_PURCHASES"] > 0):
        return("INSTALLMENTS")
    elif (x["ONEOFF_PURCHASES"] > 0) & (x["INSTALLMENTS_PURCHASES"] > 0):
        return("BOTH_ONEOFF_INSTALLMENTS")

# Process credit data for clustering
def process_credit_data(input_data):
    input_df = pd.DataFrame(input_data)

    # Feature Engineering for clustering
    input_df["Monthly_Avg_Purchase"] = input_df["PURCHASES"] / input_df["TENURE"]
    input_df["Monthly_Cash_Advance"] = input_df["CASH_ADVANCE"] / input_df["TENURE"]
    input_df["Purchase_Type"] = input_df.apply(purchasetype, axis=1)
    input_df["Balance_Credit_Ratio"] = input_df["BALANCE"] / input_df["CREDIT_LIMIT"]
    input_df["Total_Payment_Ratio"] = np.where(input_df["MINIMUM_PAYMENTS"] == 0,
                                                input_df["MINIMUM_PAYMENTS"],
                                                input_df["PAYMENTS"] / input_df["MINIMUM_PAYMENTS"])
    
    input_df = input_df.round(2)
    
    # Apply log transformation to numeric data
    input_numeric_data = input_df._get_numeric_data()
    input_df_log = input_numeric_data.apply(lambda x: np.log(x + 1))
    
    # Identify categorical variables
    input_categorical_variable_names = [x for x in list(input_df.columns) if x not in input_numeric_data.columns]
    input_categorical_variable_names.remove("CUST_ID")
    
    input_df_dummies = pd.get_dummies(input_df[input_categorical_variable_names])
    input_df_dummies = input_df_dummies.reindex(columns=dummies.columns, fill_value=0)
    input_df_dummies = input_df_dummies.astype(bool)
    
    input_df_merged = pd.concat([input_df_log, input_df_dummies], axis=1)
    
    var_names = ["BALANCE", "PURCHASES", "PAYMENTS", "MINIMUM_PAYMENTS", 
                 "PRC_FULL_PAYMENT", "TENURE", "CASH_ADVANCE", "CREDIT_LIMIT"]
    
    input_df_new = input_df_merged[[x for x in input_df_merged.columns if x not in var_names]]
    
    input_scaled = sc.transform(input_df_new)
    
    # Apply PCA transformation
    reduced_input_df = final_pc_input.transform(input_scaled)
    
    # Predict clusters
    predictions = km_4.predict(reduced_input_df)
    
    return predictions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/churn', methods=['GET', 'POST'])
def churn():
    if request.method == 'POST':
        try:
            country = request.form['country']
            credit_score = float(request.form['credit_score'])
            gender = 1 if request.form['gender'] == 'Male' else 0
            age = float(request.form['age'])
            tenure = float(request.form['tenure'])
            balance = float(request.form['balance'])
            num_products = int(request.form['num_products'])
            has_credit_card = 1 if request.form['has_credit_card'] == 'Yes' else 0
            is_active_member = 1 if request.form['is_active_member'] == 'Yes' else 0
            estimated_salary = float(request.form['estimated_salary'])

            # One-hot encode the country input (0, 1, 2 for France, Germany, Spain)
            if country == 'France':
                country_encoded = [1, 0, 0]
            elif country == 'Germany':
                country_encoded = [0, 1, 0]
            else:
                country_encoded = [0, 0, 1]

            # Create the input data array
            input_data = np.array([country_encoded + 
                                   [credit_score, gender, age, tenure, balance, num_products, 
                                    has_credit_card, is_active_member, estimated_salary]])

            # Scale the input data using the loaded scaler
            input_data_scaled = churn_scaler.transform(input_data)

            # Make the prediction
            prediction = churn_model.predict(input_data_scaled)

            result = "Will Churn" if prediction > 0.5 else "Will Not Churn"
            return render_template('churn_result.html', prediction=result)

        except Exception as e:
            return str(e)

    return render_template('churn.html')

@app.route('/loan', methods=['GET', 'POST'])
def loan():
    if request.method == 'POST':
        try:
            # Input collection from the user
            gender = 1 if request.form['gender'].lower() == 'male' else 0
            married = 1 if request.form['married'].lower() == 'yes' else 0
            dependents = int(request.form['dependents'])
            education = 1 if request.form['education'].lower() == 'graduate' else 0
            self_employed = 1 if request.form['self_employed'].lower() == 'yes' else 0
            applicant_income = float(request.form['applicant_income'])
            coapplicant_income = float(request.form['coapplicant_income'])
            loan_amount = float(request.form['loan_amount'])
            loan_amount_term = int(request.form['loan_amount_term'])
            credit_history = float(request.form['credit_history'])
            property_area = request.form['property_area'].lower()

            property_area_urban = 1 if property_area == 'urban' else 0
            property_area_semiurban = 1 if property_area == 'semiurban' else 0
            dependents_1 = 1 if dependents == 1 else 0
            dependents_2 = 1 if dependents == 2 else 0
            dependents_3_plus = 1 if dependents >= 3 else 0

            input_data = {
                'Gender': [gender],
                'Married': [married],
                'Education': [education],
                'Self_Employed': [self_employed],
                'Credit_History': [credit_history],
                'ApplicantIncomelog': [applicant_income],
                'LoanAmountlog': [loan_amount],
                'Loan_Amount_Term_log': [loan_amount_term],
                'Total_Income_log': [applicant_income + coapplicant_income],
                'Property_Area_Semiurban': [property_area_semiurban],
                'Property_Area_Urban': [property_area_urban],
                'Dependents_1': [dependents_1],
                'Dependents_2': [dependents_2],
                'Dependents_3+': [dependents_3_plus]
            }

            input_df = pd.DataFrame(input_data)

            continuous_cols = ['ApplicantIncomelog', 'LoanAmountlog', 'Loan_Amount_Term_log', 'Total_Income_log']
            input_df[continuous_cols] = np.log1p(input_df[continuous_cols])
            input_scaled = loan_scaler.transform(input_df[continuous_cols])

            input_df[continuous_cols] = input_scaled

            model_input = input_df[['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 
                                    'ApplicantIncomelog', 'LoanAmountlog', 'Loan_Amount_Term_log', 'Total_Income_log', 
                                    'Property_Area_Semiurban', 'Property_Area_Urban', 'Dependents_1', 
                                    'Dependents_2', 'Dependents_3+']] 

            prediction = rf_model.predict(model_input)

            result = "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"
            return render_template('loan_result.html', prediction=result)

        except Exception as e:
            return str(e)

    return render_template('loan.html')


@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    if request.method == 'POST':
        try:
            # Input collection from the user (this is a sample, modify as needed)
            cust_id = request.form['cust_id']
            balance = float(request.form['balance'])
            balance_frequency = float(request.form['balance_frequency'])
            purchases = float(request.form['purchases'])
            oneoff_purchases = float(request.form['oneoff_purchases'])
            installments_purchases = float(request.form['installments_purchases'])
            cash_advance = float(request.form['cash_advance'])
            purchases_frequency = float(request.form['purchases_frequency'])
            oneoff_purchases_frequency = float(request.form['oneoff_purchases_frequency'])
            purchases_installments_frequency = float(request.form['purchases_installments_frequency'])
            cash_advance_frequency = float(request.form['cash_advance_frequency'])
            cash_advance_trx = float(request.form['cash_advance_trx'])
            purchases_trx = float(request.form['purchases_trx'])
            credit_limit = float(request.form['credit_limit'])
            payments = float(request.form['payments'])
            minimum_payments = float(request.form['minimum_payments'])
            prc_full_payment = float(request.form['prc_full_payment'])
            tenure = float(request.form['tenure'])

            # Forming input data as a dictionary
            input_data = {
                'BALANCE': [balance],
                'BALANCE_FREQUENCY': [balance_frequency],
                'PURCHASES': [purchases],
                'ONEOFF_PURCHASES': [oneoff_purchases],
                'INSTALLMENTS_PURCHASES': [installments_purchases],
                'CASH_ADVANCE': [cash_advance],
                'PURCHASES_FREQUENCY': [purchases_frequency],
                'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
                'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
                'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
                'CASH_ADVANCE_TRX': [cash_advance_trx],
                'PURCHASES_TRX': [purchases_trx],
                'CREDIT_LIMIT': [credit_limit],
                'PAYMENTS': [payments],
                'MINIMUM_PAYMENTS': [minimum_payments],
                'PRC_FULL_PAYMENT': [prc_full_payment],
                'TENURE': [tenure]
            }

            input_df = pd.DataFrame(input_data)

            # Feature engineering for clustering
            input_df["Monthly_Avg_Purchase"] = input_df["PURCHASES"] / input_df["TENURE"]
            input_df["Monthly_Cash_Advance"] = input_df["CASH_ADVANCE"] / input_df["TENURE"]
            input_df["Purchase_Type"] = input_df.apply(purchasetype, axis=1)
            input_df["Balance_Credit_Ratio"] = input_df["BALANCE"] / input_df["CREDIT_LIMIT"]
            input_df["Total_Payment_Ratio"] = np.where(input_df["MINIMUM_PAYMENTS"] == 0,
                                                       input_df["MINIMUM_PAYMENTS"],
                                                       input_df["PAYMENTS"] / input_df["MINIMUM_PAYMENTS"])

            # Log transformation for numerical columns
            input_numeric_data = input_df._get_numeric_data()
            input_df_log = input_numeric_data.apply(lambda x: np.log(x + 1))

            # Categorical encoding (dummies) for "Purchase_Type"
            input_categorical_variable_names = [x for x in list(input_df.columns) if x not in input_numeric_data.columns]
            input_df_dummies = pd.get_dummies(input_df[input_categorical_variable_names])
            input_df_dummies = input_df_dummies.reindex(columns=dummies.columns, fill_value=0)
            input_df_dummies = input_df_dummies.astype(bool)

            # Merge log-transformed data with the dummies
            input_df_merged = pd.concat([input_df_log, input_df_dummies], axis=1)

            # Drop specific columns for clustering
            var_names = ["BALANCE", "PURCHASES", "PAYMENTS", "MINIMUM_PAYMENTS", 
                         "PRC_FULL_PAYMENT", "TENURE", "CASH_ADVANCE", "CREDIT_LIMIT"]
            input_df_new = input_df_merged[[x for x in input_df_merged.columns if x not in var_names]]

            # Apply standard scaling
            input_scaled = sc.transform(input_df_new)

            # Apply PCA transformation
            reduced_input_df = final_pc_input.transform(input_scaled)

            # Predict the cluster
            cluster_prediction = km_4.predict(reduced_input_df)

            # Return the predicted cluster number
            return render_template('cluster_result.html', cluster_number=cluster_prediction[0])

        except Exception as e:
            return str(e)

    return render_template('cluster.html')


if __name__ == '__main__':
    app.run(debug=True)
