# SmartBanking: ML-Powered Banking Web App

Welcome to **SmartBanking**, a comprehensive web application designed to bring machine learning insights to the banking sector. This app combines powerful predictive models with a user-friendly interface to deliver three essential services:

1. **Customer Churn Prediction**
2. **Loan Approval Prediction**
3. **Customer Clustering**

Each of these services uses a unique machine learning model fine-tuned to provide accurate, data-driven results, helping banks enhance customer retention, assess loan risks, and categorize customers effectively.

---

## Key Features

### 1. Customer Churn Prediction
   - **Model Used:** Artificial Neural Network (ANN)
   - **Purpose:** Predict which customers are likely to leave, allowing the bank to target at-risk clients with retention strategies.
   - **Data Handling:** Imbalanced datasets handled properly to improve model accuracy.

### 2. Loan Approval Prediction
   - **Model Used:** Random Forest
   - **Purpose:** Assess whether a loan application is likely to be approved, aiding in informed decision-making.
   - **Data Handling:** Log transformations and scaling techniques applied to improve predictive performance.

### 3. Customer Clustering
   - **Model Used:** K-Nearest Neighbors (KNN) after evaluating various clustering algorithms
   - **Purpose:** Segment customers into clusters based on key performance indicators (KPIs), enabling targeted marketing and personalized offers.

---

## Project Structure

The project is organized into the following directories and files:

- **`datasets/`**: Contains datasets for each of the three services.
- **`templates/`**: Includes HTML files for the web pages:
  - `home.html`: The home page
  - `churn.html` and `churn_result.html`: Pages for the Customer Churn Prediction service
  - `loan.html` and `loan_result.html`: Pages for the Loan Approval Prediction service
- **`static/`**: Contains static assets:
  - `styles.css`: Stylesheet for the website
  - `image/`: Folder containing the logo for the app
- **`notebooks/`**: Jupyter notebooks with detailed model analysis and training steps for each service
- **Root Folder**:
  - `app.py`: The main Flask application file that runs the web app
  - Model files (`*.pkl`) and scalers for each model, enabling real-time predictions

---

## Installation and Setup

To get started with **SmartBanking**, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/bluepronay/SmartBanking.git

Thanks for Visiting!!
