
import streamlit as st
import json
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Loading the dataset to get label categories
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
       background-color: #FFA07A;
    }
</style>
'''

# Applying the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Mapping dictionaries
Country_mapping = {0: 'Germany', 1: 'UK', 2: 'Australia', 3: 'Canada', 4: 'USA'}
Gender_mapping = {0: 'Male', 1: 'Female'}
Customer_Segment_mapping = {0: 'Regular', 1: 'Premium', 2: 'New'}
Product_Category_mapping = {0: 'Clothing', 1: 'Electronics', 2: 'Books', 3: 'Home Decor', 4: 'Grocery'}
Product_Brand_mapping = {0: 'Nike', 1: 'Samsung', 2: 'Penguin Books', 3: 'Home Depot', 4: 'Nestle', 5: 'Apple', 6: 'Zara', 7: 'Random House', 8: 'Coca-Cola', 9: 'Adidas',
                      10: 'Pepsi', 11: 'IKEA', 12: 'HarperCollins', 13: 'Bed Bath & Beyond', 14: 'Sony', 15: 'Whirepool', 16: 'Mitsubhisi', 17: 'BlueStar'}
Feedback_mapping = {0: 'Excellent', 1: 'Average', 2: 'Bad', 3: 'Good'}
Order_Status_mapping = {0: 'Shipped', 1: 'Processing', 2: 'Pending', 3: 'Delivered'}

# SIDEBAR 

st.sidebar.title('**Enter the features to predict the product**')

# User input
Country_n = st.sidebar.selectbox('**Country**', options=list(Country_mapping.keys()), format_func=lambda x: Country_mapping[x])
Age = st.sidebar.slider('**Age**', min_value=0, max_value=80, step=1)
Gender_n = st.sidebar.selectbox('**Gender**', options=list(Gender_mapping.keys()), format_func=lambda x: Gender_mapping[x])
Customer_Segment_n = st.sidebar.selectbox('**Customer_Segment**', options=list(Customer_Segment_mapping.keys()), format_func=lambda x: Customer_Segment_mapping[x])
Product_Brand_n = st.sidebar.selectbox('**Product Brand**', options=list(Product_Brand_mapping.keys()), format_func=lambda x: Product_Brand_mapping[x])
Product_Category_n = st.sidebar.selectbox('**Product_Category**', options=list(Product_Category_mapping.keys()), format_func=lambda x: Product_Category_mapping[x])
Feedback_n = st.sidebar.selectbox('**Feedback**', options=list(Feedback_mapping.keys()), format_func=lambda x: Feedback_mapping[x])
Order_Status_n = st.sidebar.selectbox('**Order_Status**', options=list(Order_Status_mapping.keys()), format_func=lambda x: Order_Status_mapping[x])
Ratings = st.sidebar.slider('**Ratings**', min_value=0.0, max_value=5.0, step=0.1)


# Prediction
if st.sidebar.button('**PREDICT**'):
   features = [
       Country_n, Age, Gender_n, Customer_Segment_n,
       Product_Brand_n, Product_Category_n, Feedback_n,
       Order_Status_n, Ratings
   ]
   prediction_n = knn.predict([features])[0]


   if prediction_n in products_mapping:
       prediction = products_mapping[prediction_n]
       st.sidebar.write(f'The predicted product is: {prediction}')
   else:
       st.sidebar.write("Error: The predicted value does not correspond to any known product.")

# MAIN TEXT

st.header("PRODUCT RECOMMENDATION PREDICTOR")
col1, col2 = st.columns([2,2])

with col1:
        st.image('https://novasolutions.ca/wp-content/uploads/2021/01/personalized-product-recommendations1.jpg', width=400, use_column_width = 'auto')

with col2:
        st.write('''
        **Problem**: Customers often need help finding products that match their preferences and needs. These can be complementary products that are often bought in addition to the selected product, or related products that are similar to a selected product. Additionally, potential substitutes are similar products that customers might like.
        **Solution**: A product recommendation application that suggests products based on the user's purchase history and preferences.
           ''')

with st.expander("**INSTRUCTIONS**"):
        st.write("**Step 1**: Enter parameters: On the left side of the screen you will find a sidebar with input options.")
        st.write("**Step 2**: Make the prediction: Once you have entered all the parameters, click on the 'Predict' button. The application will display the prediction result on the screen.")

with st.container():
        st.write("**Hope you enjoy using the PRODUCT RECOMMENDATION PREDICTOR 😊**")

st.divider()

# TABLES

tab1, tab2, tab3 = st.tabs(["**DATASET DESCRIPTION and EDA**", "**MACHINE LEARNING MODEL**", "**CONCLUSIONS**"])

tab1.write("**DATASET DESCRIPTION and EDA**")
tab1.write('''
        The dataset represents retail transactional data. It contains information about customers, their purchases, products, and transaction details.
        It was downloaded from the Kaggle web page: https://www.kaggle.com/datasets/sahilprajapati143/retail-analysis-large-dataset"
           ''')


tab1.write('''
        **Analysis of the descriptive statistical variables shows**:
           
        - There is a wide range of ages among the customers.
        - Variability in the amounts spent among the customers.
        - Variability in the ratings of products purchased.           
''')


tab1.write('''
        **Univariate analysis of (categorical) variables reveals**:
           
        - The most popular item to order is water (spring, boottled, artesian and distilled) while the least popular item is package AC.
        - The majority of the customers are from the USA; UK and Germany are the next most popular countries, and the smallest groups of customers are from Australia and Canada. 
        - There are almost twice as many men as women making transactions.
        - Most transactions correspond to the middle-income customers, whereas low income-customers are the second group and high income-customers tend to  make less transactions.
        - The biggest group of customers represents the regular customer segment, almost half of the customers belong to a new segment, whereas the premium segment is the smallest group.
        - There are five categories, being the two most popular electronics and grocery; clothing, books and home decor are equally in demand.
        - Pepsi is the most common product brand, the rest of the brands are equally in demand with the least demanded brands such as: Whirepool, Mitsubhisi and BlueStar.
        - Most of the customers are satisfied with a product or service, however, there are some that left a negative feedback.
        - There is no big difference in the shipping methods, being "same-day" and "express" shipping slightly more popular than the "standard" method.
        - People use more credit cards than debit.
        - Orders are mainly delivered, however, there are some of them with the status shipped, processing or pending, that likely depends on the date of order.
   ''')

tab1.write("**Graphs**:")
with tab1:
   st.image("/workspaces/Product-recommendation-final-project/output.corr.png", use_column_width = 'auto', caption ="Univariate analysis of (categorical) variables")


tab1.write('''
        **Correlation analysis displays**:

        - There is a negative relationship between **Country and Customer Age** as well as between **Country and Income**, thus, in certain countries customers are more likely to have a high income than in other countries, the same with the Age, customers of one country tend to be younger than customers of others. 
        - We can observe a positive relationship between **Age and Customer Segment**, which makes a lot of sense, the older the customer the higher the probability that he/she belongs to the premium segment.
        - There is a negative relationship between **Age and Feedback**, and positive relationship between Age and Ratings, thus, older customers tend to leave more positive feedback and give higher ratings.
        - **Customer Segment with Ratings** have a positive relationship: newly arrived customers are likely to receive better service, as a marketing ploy to attract customers and convert them into their clients, as a result, they are more satisfied giving better ratings. Here also can be included **Order Status** for different customer segments.
        - **Product Brand correlates with Feedback, Order Status and Ratings** in a positive way. Famous and recognised brands, such as Apple, Adidas, etc. have good reputation with mostly positive feedback and high ratings. Moreover, they try to deliver the best customer service (f.e. on-time delivery). 
        - There is a negative relationship between **Feedback and Ratings**, thus, the higher the ratings the more positive feedback the product receives.
        - The rest of the correlations remain the same as previously seen.

           ''')


tab1.write("**Graphs**:")
with tab1:
   st.image("/workspaces/Product-recommendation-final-project/output.png", use_column_width = 'auto', caption ="Correlation analysis")


tab1.write('''
        Through a feature importance analysis and the model performance analysis, the following **8 optimal features** (user input in the app) were selected:
        Age, Gender, Customer Segment, Product Brand, Product Category, Feedback, Order status, Ratings
           ''')



tab2.write("**MACHINE LEARNING MODEL**")
tab2.write('''
           After training several models, **K-NEAREST NEIGHBOURS model** was selected as the best one. 
           KNN is a supervised learning algorithm that can be used for classification. 
           In the context of recommendation systems, KNN is commonly used to find similar products based on certain characteristics. 
           The goal is to recommend products that are similar to those the user has viewed or purchased previously.
           The recommendation algorithm is based on a machine learning system that analyzes data to determine user preferences and suggest products that they might genuinely be interested in.
           ''') 
tab2.write(''' 
           The performance metrics (Accuracy: 0.98; Precision: 0.97; Recall: 0.96; F1-Score: 0.965) indicate that the KNN model is highly effective in recommending relevant products to users.
           ''') 

tab3.write("**CONCLUSIONS**")
tab3.write('''
           Providing effective recommendations to your customers is a key way to enhance the end-user experience, retain customers, and keep them happy. 
           Thus, a product recommendation application that suggests products based on the user's purchase history and preferences, could be a perfect tool.
           ''')
tab3.write('''
           **Advantages**:
           Interactive and user-friendly interface.
           Fast and personalized results.

           **Possible disadvantages**:
           The limitation of relying on a rating system is that products without user ratings won't be able to give a whole picture. 
           Therefore, you would have to force recommendations through push surveys or add some fictitious ratings to an unrated product in the system. 
           Another drawback is that this type of algorithm tends to form groups of similar users who, over time, will receive the same recommendations, 
           meaning the suggested products will become less personalized over time.
           ''')


  