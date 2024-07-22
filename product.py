
import streamlit as st
import json
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Loading the dataset to get label categories
df = pd.read_csv('new_retail_data.csv') 


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
                 **Problem**: Customers frequently require assistance in discovering products that align with their preferences and needs. 
                 These could include complementary products commonly purchased alongside the selected product, related products that are similar 
                 to the selected item, or potential substitutes that might appeal to customers.

                 **Solution**: A product recommendation application that provides personalized suggestions based on the user's purchase history and preferences.
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
        **Discovering the Trends: What Do Customers Really Like?**
           
There are certain trends and preferences that stand out as a result of analysis:

- **Water: The Star of the Show**
In a surprising twist, the most popular item flying off the shelves isn't a fancy gadget or a trendy fashion item—it's water! From spring to bottled, artesian to distilled, everyone wants a sip. 
On the flip side, poor package AC sits lonely on the shelf, barely catching anyone's eye.

- **Distribution of Customers by their Countries**
The majority of purchases come from the USA, making it the largest group of buyers. Close behind are customers from the UK and Germany, showing significant engagement. 
Interestingly, only a few transactions originate from Australia and Canada, marking these countries as the least represented. 
This diverse distribution of buyers highlights the global reach and appeal of the marketplace, with each purchase contributing to a vibrant, interconnected economy.

- **The Gender Balance**
Men dominate the scene, making nearly twice as many transactions as women. This dynamic adds an intriguing layer to the shopping patterns observed.

- **Income Insights**
Most shoppers belong to the middle-income bracket, swarming the stalls with their purchases. Low-income shoppers come in next, while high-income customers seem to stroll leisurely, making fewer transactions.

- **Customer Segments: Who's Who?**
The largest group consists of regular customers, while a significant portion is new, exploring the marketplace with fresh eyes. The premium segment is the most exclusive, with only a select few belonging to this elite group.

- **Category Craze**
The marketplace offers five categories. Electronics and groceries are the stars of the show, attracting the most attention. Clothing, books, and home decor are equally popular, creating a balanced demand.

- **Brand Battles**
Pepsi reigns supreme as the most common product brand. Other brands hold their ground, but Whirepool, Mitsubhisi, and BlueStar lag behind, barely catching the shoppers' fancy.

- **Customer Satisfaction: The Good and the Bad**
Most customers leave with a smile, satisfied with their purchase or service. Yet, a few grumble, leaving negative feedback that stands out in the sea of positive reviews.

- **Shipping Preferences**
When it comes to shipping, there's no huge difference. "Same-day" and "express" methods are slightly favored over the "standard" option, showing a slight preference for speed.

- **Payment Methods**
Credit cards are the champions, with more people swiping them than using debit cards.

- **Order Status: The Waiting Game**
While most orders are delivered promptly, some linger in statuses like shipped, processing, or pending, possibly due to the order date.
   ''')

tab1.write("**Graphs**:")
with tab1:
   st.image("/workspaces/Product-recommendation-final-project/output.png", use_column_width = 'auto', caption ="Univariate analysis of (categorical) variables")


tab1.write('''
        **Unveiling Hidden Patterns: A Curious Look at Correlation Analysis**

As you sift through the data, some fascinating connections come to light:

- **The Country Conundrum**
The initial findings show that there is a negative relationship between a customer's Country and their Age, as well as between their Country and Income. In certain countries, customers are more likely to be wealthier and younger compared to others. It's like finding out that in some lands, youthful riches are more common!

- **The Age Advantage**
Moving on, we uncover a positive relationship between Age and Customer Segment. This makes perfect sense: as customers age, they are more likely to climb into the premium segment. It's as if loyalty and spending power grow stronger with age.

- **Feedback Frequencies**
Digging deeper, we notice that Age negatively correlates with Feedback, but positively with Ratings. Older customers tend to leave glowing reviews and high ratings. Perhaps wisdom comes with age, leading to more positive shopping experiences.

- **Segment Strategies**
The relationship between Customer Segment and Ratings is positively intriguing. Newcomers often receive better service, possibly as a marketing strategy to win them over. 
Satisfied new customers leave higher ratings, helping businesses turn them into loyal clients. Order Status plays a role here too, with different segments experiencing varied service levels.

- **Brand Brilliance**
Our investigation reveals that Product Brand positively correlates with Feedback, Order Status, and Ratings. Renowned brands like Apple and Adidas, with their stellar reputations, tend to receive positive feedback and high ratings. 
They ensure top-notch customer service, including on-time deliveries, to maintain their standing.

- **Feedback Formula**
There's a negative relationship between Feedback and Ratings. The higher the ratings, the more positive the feedback. It's a simple yet powerful connection: happy customers sing praises.

- **Consistent Correlations**
Lastly, most of the other correlations follow previously seen patterns, maintaining a steady rhythm in the background.

           ''')


tab1.write("**Graphs**:")
with tab1:
   st.image("/workspaces/Product-recommendation-final-project/output.corr.png", use_column_width = 'auto', caption ="Correlation analysis")


tab1.write('''
        Based on a comprehensive feature importance analysis and model performance evaluation, the optimal features were identified and selected. These features are:

        - Country
        - Age
        - Gender
        - Customer Segment
        - Product Brand
        - Product Category
        - Feedback
        - Order Status
        - Ratings
                
        These features represent the most significant factors influencing the model's predictive accuracy and overall performance.
           ''')



tab2.write("**MACHINE LEARNING MODEL**")
tab2.write('''
           After training several models, **K-NEAREST NEIGHBOURS model** was selected as the best one. 
           KNN is a supervised learning algorithm that can be used for classification. 
           In the context of recommendation systems, KNN is commonly used to find similar products based on certain characteristics. 
           The goal is to recommend products that are similar to those the user has purchased previously.
           The recommendation algorithm is based on a machine learning system that analyzes data to determine user preferences and suggest products that they might genuinely be interested in.
           ''') 

with tab2:
      st.image("https://miro.medium.com/v2/resize:fit:1151/0*ItVKiyx2F3ZU8zV5", use_column_width = 'auto', caption ="K-NEAREST NEIGHBOURS model")

tab2.write(''' 
           The performance metrics (Accuracy: 0.98; Precision: 0.97; Recall: 0.96; F1-Score: 0.965) indicate that the KNN model is highly effective in recommending relevant products to users.
           ''') 

tab3.write("**CONCLUSIONS**")
tab3.write('''
           Providing effective recommendations to your customers is essential for enhancing the end-user experience, retaining customers, and maintaining their satisfaction. A product recommendation application, which suggests products based on the user's purchase history and preferences, can serve as an invaluable tool.

        **Advantages**:

        - **Interactive and User-Friendly Interface**: The application offers an intuitive and engaging interface, ensuring ease of use for all customers.
        Fast and Personalized Results: It delivers quick and tailored product recommendations, enhancing the overall shopping experience.

        **Potential Disadvantages**:

        - **Dependence on a Rating System**: Products lacking user ratings may not provide a complete picture. 
        To address this, you might need to prompt recommendations via push surveys or introduce some hypothetical ratings for unrated products within the system.
        - **Homogenization of Recommendations**: Over time, the algorithm may create clusters of similar users, leading to repetitive recommendations for these groups. 
        This could result in the suggested products becoming less personalized, diminishing the uniqueness of the recommendations.
           ''')


  