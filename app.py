import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Create and fit the OneHotEncoder on relevant training features
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoder.fit(df[['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']])  # Fit on relevant columns

st.title("Laptop Predictor")

# Collect input features from the user
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Query preparation
    ppi = None
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create the query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Reshape and create DataFrame for the input query
    query_df = pd.DataFrame(query.reshape(1, -1), columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'Ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    # Encode the categorical features only
    encoded_query = encoder.transform(query_df[['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']])  # Use only relevant columns

    # Combine the encoded features with the numeric features
    numeric_features = query_df[['Ram', 'Weight', 'Touchscreen', 'Ips', 'Ppi', 'HDD', 'SSD']].values
    final_query = np.hstack((encoded_query, numeric_features))

    # Print shapes for debugging
    print("Encoded query shape:", encoded_query.shape)  # Should show the shape of the encoded features
    print("Numeric features shape:", numeric_features.shape)  # Should show the shape of numeric features
    print("Final query shape:", final_query.shape)  # Should match the number of features the model expects

    # Make prediction
    predicted_price = pipe.predict(final_query)
    st.title("The predicted price of this configuration is " + str(int(np.exp(predicted_price[0]))))
