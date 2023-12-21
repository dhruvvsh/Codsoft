import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load pre-trained model
with open('Ratings_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load encoded column names
with open('encoded_columns.pkl', 'rb') as columns_file:
    encoded_columns = pickle.load(columns_file)

# Load predictions from the file
#with open('predictions.pkl', 'rb') as file:
    #predictions = pickle.load(file)

# Function to preprocess user input
def preprocess_input(user_input):
    # Perform any necessary preprocessing based on your training data preprocessing steps
    # Make sure to handle categorical variables, convert types, and match the feature format used during training
    # For simplicity, assuming user_input is a dictionary with keys 'Year' and 'Votes'
    df= pd.DataFrame(user_input, index=[0])

 
    # One-hot encode categorical features using the same columns as during training
    df_encoded = pd.get_dummies(df, columns=['Genre', 'Director', 'Actors','Name','Year'])
    df_encoded = df_encoded.reindex(columns=encoded_columns, fill_value=0)

    return df_encoded


# Streamlit App
def main():
    st.title("Movie Rating Prediction App")

    # User Input Form
    Name = st.text_input("Enter Movie Name")
    Genre = st.text_input("Enter Movie Genre")
    Director = st.text_input("Enter Director's Name")
    Actors = st.text_area("Enter Actors (separated by commas)")
    Year = st.text_input("Enter the Year")

   

    # Make Prediction Button
    if st.button("Predict Rating"):
        # Preprocess user input
        user_input = preprocess_input({'Genre': [Genre], 'Director': [Director], 'Actors': [Actors],'Name':[Name],'Year':[Year]})
        
        # Make a prediction using the model
        predicted_rating = model.predict(user_input)
        
        # Display the prediction
        st.success(f"Predicted Rating: {predicted_rating[0]:.1f}")
        
        # Display saved predictions
       #st.info(f"Saved Predictions: {predictions}")

if __name__ == '__main__':
    main()
