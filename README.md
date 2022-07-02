# Cat Prediction
A web-app that predicts the likely location of Snowy the cat

Use the sliders and selection boxes to update feature values, such as day and temperature.

A classification model, trained with 3 months of historic behaviour data, is used to predict the most likely location for Snowy

## Streamlit

This is the code for the web-application

You can see a live version of this web application  [here](https://cat-predict-app.herokuapp.com/) and the blog is [here](https://simon-aubury.medium.com/can-ml-predict-where-my-cat-is-now-part-2-7efaec267339)

## Data and training
This repo is only the web-application. The code to train a model is [here](https://github.com/saubury/cat-predict)

## Setup


Ensure Python 3, `virtualenv` and `pip` are installed.

```console
which python3

virtualenv -p `which python3` venv
source venv/bin/activate
python --version
pip --version
pip install -r requirements.txt 
```


## Streamlit


```console
streamlit run cat_predictor_app.py
```
