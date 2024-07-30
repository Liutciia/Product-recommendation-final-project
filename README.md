# Product Recommendation Predictor


## 📚 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Contact](#contact)

  
## 🌟 Introduction

Welcome to the Product Recommendation Predictor! This project aims to build a recommendation engine using the K-Nearest Neighbors (KNN) algorithm to suggest products to users based on their preferences and historical data.


## 📊 Dataset

The dataset used for this project includes information on user purchases, ratings, and product details. The dataset contains the following features:

Transaction_ID: Unique identifier for each transaction.
Customer_ID: Unique identifier for each customer.
Name: Name of the customer.
Email: Email address of the customer.
Phone: Phone number of the customer.
Address: Address of the customer.
City: City where the customer resides.
State: State where the customer resides.
Zipcode: Zip code of the customer's address.
Country: Country where the customer resides.
Age: Age of the customer.
Gender: Gender of the customer.
Income: Income level of the customer.
Customer_Segment: Segment of the customer (e.g., Premium, Regular, New).
Year: Year component extracted from the last purchase date.
Month: Month component extracted from the last purchase date.
Date: Date component extracted from the last purchase date.
Time: Time component extracted from the last purchase date.
Total_Purchases: Total number of purchases made by the customer.
Amount: Amount spent in a single transaction.
Total_Purchase_Amount: Total amount spent by the customer (calculated as Amount * Total_Purchases).
Product_Category: Category of the purchased product.
Product_Brand: Brand of the purchased product.
Product_Type: Type of the purchased product.
Feedback: Feedback provided by the customer on the purchase.
Shipping_Method: Method used for shipping the product.
Payment_Method: Method used for payment.
Order_Status: Status of the order (e.g., Pending, Processing, Shipped, Delivered).
Ratings : ratings given by customers on different products.
products: list of different products.


## 🛠 Methodology

**Data Preprocessing**

Cleaning: Remove missing values, duplicates and irrelevant information.
Encoding: Convert categorical features to numerical using techniques like factorizing.
Feature scaling: Scale features so that they fall within a specific range [0, 1].

**Model Training and Optimization**

- Feature Selection: Select relevant features for the KNN model.
- Splitting Data: Split the data into training and testing sets.
- Training: Train the KNN model on the training data.
- Optimization: Optimize the model using hyperparameters.


## 🏆 Results

The KNN model achieved the following performance metrics:

- **Accuracy**: 0.98
- **Precision**: 0.97
- **Recall**: 0.96
- **F1-Score**: 0.965
  
These results indicate that the KNN model is highly effective in recommending relevant products to users.


## 🚀 Usage

To try this recommendation system, you can use the following Interactive Streamlit App:

https://product-recommendation-predictor.onrender.com


## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- streamlit 


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss any changes.


## 📞 Contact
For any questions or suggestions, feel free to contact me:

- **[GitHub](https://github.com/Liutciia)** 
