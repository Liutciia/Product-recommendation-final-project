
import streamlit as st
import json
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Loading the dataset
df = pd.read_csv('/workspaces/Product-recommendation-final-project/src/new_retail_data.csv') 


# Generating product mapping dictionaries
unique_products = df['products'].unique()
products_mapping = {i: product for i, product in enumerate(unique_products)}
products_reverse_mapping = {product: i for i, product in enumerate(unique_products)}


# Loading model
knn = joblib.load('knn_classifier_default.sav')


# Streamlit app
# Defining el CSS para el fondo con imagen
page_bg_img = '''
<style>
   .stApp {
       background-image: url("https://novasolutions.ca/wp-content/uploads/2021/01/personalized-product-recommendations1.jpg");
       background-size: cover;
   }
</style>
'''


# Applying the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


st.title('Product Recommendation Predictor')
st.write('Enter the features to predict the product.')


# Mapping dictionaries
Country_mapping = {0: 'Germany', 1: 'UK', 2: 'Australia', 3: 'Canada', 4: 'USA'}
Gender_mapping = {0: 'Male', 1: 'Female'}
Customer_Segment_mapping = {0: 'Regular', 1: 'Premium', 2: 'New'}
Product_Category_mapping = {0: 'Clothing', 1: 'Electronics', 2: 'Books', 3: 'Home Decor', 4: 'Grocery'}
Product_Brand_mapping = {0: 'Nike', 1: 'Samsung', 2: 'Penguin Books', 3: 'Home Depot', 4: 'Nestle', 5: 'Apple', 6: 'Zara', 7: 'Random House', 8: 'Coca-Cola', 9: 'Adidas',
                      10: 'Pepsi', 11: 'IKEA', 12: 'HarperCollins', 13: 'Bed Bath & Beyond', 14: 'Sony', 15: 'Whirepool', 16: 'Mitsubhisi', 17: 'BlueStar'}
Feedback_mapping = {0: 'Excellent', 1: 'Average', 2: 'Bad', 3: 'Good'}
Order_Status_mapping = {0: 'Shipped', 1: 'Processing', 2: 'Pending', 3: 'Delivered'}


# User input
Country_n = st.selectbox('Country', options=list(Country_mapping.keys()), format_func=lambda x: Country_mapping[x])
Age = st.slider('Age', min_value=0, max_value=80, step=1)
Gender_n = st.selectbox('Gender', options=list(Gender_mapping.keys()), format_func=lambda x: Gender_mapping[x])
Customer_Segment_n = st.selectbox('Customer_Segment', options=list(Customer_Segment_mapping.keys()), format_func=lambda x: Customer_Segment_mapping[x])
Product_Brand_n = st.selectbox('Product Brand', options=list(Product_Brand_mapping.keys()), format_func=lambda x: Product_Brand_mapping[x])
Product_Category_n = st.selectbox('Product_Category', options=list(Product_Category_mapping.keys()), format_func=lambda x: Product_Category_mapping[x])
Feedback_n = st.selectbox('Feedback', options=list(Feedback_mapping.keys()), format_func=lambda x: Feedback_mapping[x])
Order_Status_n = st.selectbox('Order_Status', options=list(Order_Status_mapping.keys()), format_func=lambda x: Order_Status_mapping[x])
Ratings = st.slider('Ratings', min_value=0.0, max_value=5.0, step=0.1)


# Prediction
if st.button('Predict'):
   features = [
       Country_n, Age, Gender_n, Customer_Segment_n,
       Product_Brand_n, Product_Category_n, Feedback_n,
       Order_Status_n, Ratings
   ]
   prediction_n = knn.predict([features])[0]


   if prediction_n in products_mapping:
       prediction = products_mapping[prediction_n]
       st.write(f'The predicted product is: {prediction}')
   else:
       st.write("Error: The predicted value does not correspond to any known product.")
