import streamlit as st 
import pickle 
import pandas as pd 
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder 

@st.cache
def load_data():
    DATA_PATH = "Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)
    return data
@st.cache
def load_steps():
	steps = pd.read_csv("steps.csv")
	return steps

steps = load_steps()
data=load_data()

def load_model():
    with open('disease.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model=load_model()

encoder=model['encoder']
final_svm_model=model['svm']
final_nb_model=model['nb']
final_rf_model=model['rf']



X = data.iloc[:,:-1]
y = data.iloc[:, -1]

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}
num=st.number_input("enter the number of symptoms")
input_list=[]
for i in range(int(num)):
    s=st.selectbox("select a symptom",symptom_index.keys(),key=i)
    input_list.append(s)

def predictDisease(symptoms):
	
	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": nb_prediction,
		"final_prediction":final_prediction
	}
	step = steps.loc[steps['Disease']==final_prediction,'Steps'].item()
	return predictions,step
ok=st.button('predict')
if ok:
	pred,step = predictDisease(input_list)
	st.write('Random Forest Model : '+pred['rf_model_prediction'])
	st.write('Naive Bayes Model : '+pred['naive_bayes_prediction'])
	st.write('SVM  Model : '+pred['svm_model_prediction'])
	st.write('Final Prediction : '+pred['final_prediction'])
	st.write(step)