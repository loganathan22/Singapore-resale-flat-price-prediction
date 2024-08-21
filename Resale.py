# Packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

def town_mapping(town_map):
    town_dict = {
        'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
        'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
        'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
        'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17,
        'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21, 'TAMPINES': 22,
        'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
    }
    return town_dict.get(town_map, -1)

def flat_type_mapping(flt_type):
    flat_type_dict = {
        '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, 'EXECUTIVE': 5,
        'MULTI-GENERATION': 6
    }
    return flat_type_dict.get(flt_type, -1)

def flat_model_mapping(fl_m):
    flat_model_dict = {
        '2-room': 0, '3Gen': 1, 'Adjoined flat': 2, 'Apartment': 3, 'DBSS': 4, 'Improved': 5,
        'Improved-Maisonette': 6, 'Maisonette': 7, 'Model A': 8, 'Model A-Maisonette': 9,
        'Model A2': 10, 'Multi Generation': 11, 'New Generation': 12, 'Premium Apartment': 13,
        'Premium Apartment Loft': 14, 'Premium Maisonette': 15, 'Simplified': 16, 'Standard': 17,
        'Terrace': 18, 'Type S1': 19, 'Type S2': 20
    }
    return flat_model_dict.get(fl_m, -1)

def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt):
    year_1 = int(year)
    town_2 = town_mapping(town)
    flt_ty_2 = flat_type_mapping(flat_type)
    flr_ar_sqm_1 = int(flr_area_sqm)
    flt_model_2 = flat_model_mapping(flat_model)
    
    # Ensure storey values are positive and greater than zero for logarithm calculation
    if stry_start <= 0 or stry_end <= 0:
        raise ValueError("Storey start and end must be positive non-zero values.")
    
    str_str = np.log(int(stry_start))
    str_end = np.log(int(stry_end))
    rem_les_year = int(re_les_year)
    rem_les_month = int(re_les_month)
    lese_coms_dt = int(les_coms_dt)

    with open(r"Resale_Flat_Prices_Model_1.pkl", "rb") as f:
        regg_model = pickle.load(f)

    user_data = np.array([[year_1, town_2, flt_ty_2, flr_ar_sqm_1, flt_model_2, str_str, str_end, rem_les_year, rem_les_month, lese_coms_dt]])
    y_pred_1 = regg_model.predict(user_data)
    price = np.exp(y_pred_1[0])

    return round(price)

st.set_page_config(layout="wide")

st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
st.write("")

with st.sidebar:
    select = option_menu("MAIN MENU", ["Home", "Price Prediction", "About"])

if select == "Home":
    st.header("HDB Flats:")
    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')
    
    st.header("Resale Process:")
    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')
    
    st.header("Valuation:")
    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')
    
    st.header("Eligibility Criteria:")
    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")
    
    st.header("Resale Levy:")
    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")
    
    st.header("Grant Schemes:")
    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")
    
    st.header("HDB Loan and Bank Loan:")
    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")
    
    st.header("Market Trends:")
    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")
    
    st.header("Online Platforms:")
    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")

elif select == "Price Prediction":
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Select the Year", [str(i) for i in range(2015, 2025)])
        
        town = st.selectbox("Select the Town", [
            'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
            'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
            'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
            'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
            'TOA PAYOH', 'WOODLANDS', 'YISHUN'
        ])
        
        flat_type = st.selectbox("Select the Flat Type", [
            '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
        ])
        
        flr_area_sqm = st.number_input("Enter the Value of Floor Area sqm (Min: 31 / Max: 280)")

        flat_model = st.selectbox("Select the Flat Model", [
            'Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'
        ])
        
    with col2:
        stry_start = st.number_input("Enter the Value of Storey Start", min_value=1, value=1)
        stry_end = st.number_input("Enter the Value of Storey End", min_value=1, value=1)
        re_les_year = st.number_input("Enter the Value of Remaining Lease Year (Min: 42 / Max: 97)", min_value=42, max_value=97, value=42)
        re_les_month = st.number_input("Enter the Value of Remaining Lease Month (Min: 0 / Max: 11)", min_value=0, max_value=11, value=0)
        les_coms_dt = st.selectbox("Select the Lease Comencement Date", [str(i) for i in range(1967, 2024)])

    if st.button("Click Here to Predict"):
        try:
            pre_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt)
            st.success(f"The predicted price of the flat is: ${pre_price}")
        except ValueError as e:
            st.error(f"Error in prediction: {e}")

elif select == "About":
    st.header("Resale Flat Prices Prediction Tool")
    st.write("This tool predicts the resale prices of HDB flats in Singapore using machine learning algorithms.")
    st.header(":blue[Features:]")
    st.write("1. Year: Select the year of the resale transaction.")
    st.write("2. Town: Select the town where the flat is located.")
    st.write("3. Flat Type: Select the type of the flat (e.g., 4-room, 5-room).")
    st.write("4. Floor Area: Enter the floor area of the flat in square meters.")
    st.write("5. Flat Model: Select the model of the flat (e.g., Improved, New Generation).")
    st.write("6. Storey Range: Enter the storey range (start and end) of the flat.")
    st.write("7. Remaining Lease: Enter the remaining lease period in years and months.")
    st.write("8. Lease Commencement Date: Select the lease commencement date of the flat.")

    st.header(":blue[Machine Learning Model:]")
    st.write("The prediction tool uses a machine learning model trained on historical HDB resale transaction data. The model takes into account various features such as the flat type, floor area, storey range, and remaining lease to provide an estimated resale price.")

    st.header(":blue[Data Preprocessing:]")
    st.write("Before training the model, the data is preprocessed to handle missing values, encode categorical features, and normalize numerical features. This ensures that the model can make accurate predictions.")

    st.header(":blue[Feature Engineering:]")
    st.write("Create relevant features for the machine learning model. For example, extract numerical features such as flat type, floor area, remaining lease, storey range, and town. Normalize and scale the features to ensure they are in a suitable format for modeling.")

    st.header(":blue[Model Selection:]")
    st.write("Choose an appropriate machine learning algorithm for the task. Common choices include linear regression, decision trees, or more complex models like Random Forest or Gradient Boosting. Train the model using the preprocessed dataset.")

    st.header(":blue[Model Training:]")
    st.write("Split the dataset into training and testing sets. Train the chosen model on the training data and validate its performance on the testing data. Use cross-validation techniques to fine-tune hyperparameters and avoid overfitting.")

    st.header(":blue[Model Evaluation:]")
    st.write("Evaluate the performance of the trained model using relevant metrics such as mean absolute error (MAE), root mean square error (RMSE), and R-squared. Analyze the model's performance to ensure it meets the desired accuracy and reliability.")

    st.header(":blue[Prediction and Deployment:]")
    st.write("Once the model is trained and evaluated, deploy it in a user-friendly tool for price prediction. Create a web application or API where users can input the relevant features of a resale flat and obtain an estimated resale price based on the trained model.")

    st.header(":blue[Visualization and Insights:]")
    st.write("Incorporate visualizations and insights into the tool to provide users with a better understanding of the factors influencing resale flat prices. Display trends, patterns, and comparisons to help users make informed decisions.")

    st.header(":blue[Continuous Improvement:]")
    st.write("Monitor the performance of the deployed model and update it periodically with new data to ensure its accuracy and relevance. Incorporate user feedback and make necessary improvements to enhance the tool's usability and functionality.")