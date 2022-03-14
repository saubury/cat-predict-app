import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Settings
daysList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', ]
labelsList = ['bedroom', 'dining', 'lounge', 'nicholas', 'outside', 'study', 'winter_garden', ]

@st.cache(allow_output_mutation=True)
def prepClassification():
    ret = pickle.load(open('./cat_predictor_app.pkl', 'rb'))
    print("Classification loaded")
    return ret

@st.cache(allow_output_mutation=True)
def prepImageData():
    imageData = []
    for i in labelsList:
        # print(i)
        imageFileName = './images/{}.jpg'.format(i)
        image = Image.open(imageFileName)  
        imageData.append(image)
    print("Images loaded")
    return imageData

# Reads in saved classification model
load_clf = prepClassification()
load_image = prepImageData()

# Main page render
st.write("""
# Cat Prediction App!
A web-app that predicts the likely location of Snowy the cat - by [Simon Aubury](https://twitter.com/simonaubury)

Use the sliders and selection boxes to update feature values, such as day and temperature.

A classification model, trained with 3 months of historic behaviour data, is used to predict the most likely location for Snowy
""")


IndoorTemp = st.sidebar.slider('Indoor Temp C', 16 , 28, 22)
OutdoorTemp = st.sidebar.slider('Outdoor Temp C', 10 , 30, 22)
HourOfDay = st.sidebar.slider('Hour of day', 0 , 23, 8)
DayOfWeek = st.sidebar.selectbox('Day Of Week', daysList)
IsRaining = st.sidebar.selectbox('Raining',('True', 'False'), 1)


with st.expander("See explanation"):
     st.write("""
Machine learning are algorithms that learn from examples. I wanted to build a ML model to predict where my cat Snowy was likely to go knowing the temperature and time. 
You can use this website to predict where she is likely to be by moving the sliders around on the left.
     """)
     st.image('images/prediction.jpg')
     st.write("""
This prectiction uses a Random Forest classification - a predictive model that assigns a class label to inputs, based on many examples it has been trained on from 
thousands of past observations of time of day, temperature and location.

If you would like to learn more, have a look at the [blog](https://simon-aubury.medium.com/can-ml-predict-where-my-cat-is-now-part-1-cfb194b51aab) or the [code on github](https://github.com/saubury/cat-predict-app).
     """)

# Try a prediction
data = {'indoor_temp':[IndoorTemp], 
        'outside_temp':[OutdoorTemp], 
        'hr_of_day':[HourOfDay], 
        'day_of_week':[daysList.index(DayOfWeek)], 
        'is_raining':[IsRaining=='True'],}
pdf = pd.DataFrame(data)


# Apply model to make predictions
prediction = load_clf.predict(pdf)
prediction_proba = load_clf.predict_proba(pdf)
predictionText = load_clf.predict(pdf)[0]




# Write text prediction
st.subheader('Prediction')
st.write('Prediction : **{}**'.format(predictionText))

# Update image prediction
image = load_image[labelsList.index(predictionText)]
st.image(image)

# Show prediction probability table
with st.expander("Detailed probability table"):
    st.subheader('Prediction probability by class')
    st.dataframe(pd.DataFrame(prediction_proba, columns=load_clf.classes_)) 


