import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import streamlit as st 

st.header("DIABETES AWARENESS")
st.sidebar.title("Diabetes prediction using machine learning")
st.sidebar.image("C:\\Users\\om\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-09-06 085651.png")
scaler = StandardScaler()
data = pd.read_csv('diabetes.csv')
#print(data)
shape = data.shape
#print(shape)
describe = data.describe()
#print(describe)

X = data.drop(columns = 'Outcome')
Y = data['Outcome']
#print(Y)
standard_scaler = scaler.fit_transform(X)
#print(standard_scaler)

x_train , x_test,y_train,y_test = train_test_split (standard_scaler,Y,random_state = 2,test_size= 0.2,stratify = Y)
#print(X.shape , x_train.shape ,x_test.shape )
''' Please enter here your record for prediction'''

classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)

predicted = classifier.predict(x_train)
#print(predicted)


predicted2 = classifier.predict(x_test)
#print(predicted2)

print("accuracy score accordingly training data :", accuracy_score(y_train,predicted))
print("accuracy score accordingly testing data :", accuracy_score(y_test,predicted2))

'''making a predictive system'''

Pregnancies = st.number_input("Pregnancies : ")
Glucose = st.number_input("Glucose : ")
BloodPressure = st.number_input("BloodPressure : ")
SkinThickness = st.number_input("SkinThickness : ")
Insulin = st.number_input("Insulin : ")
BMI = st.number_input("BMI : ")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction : ")
Age = st.number_input("Age : ")
record = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)

if st.button("Predict"):
    input_as_array = np.asarray(record)
    input_as_array_reshaped = input_as_array.reshape(1,-1)
    #print(input_as_array_reshaped)

    standard_scale = scaler.transform(input_as_array_reshaped)
    input_predicted = classifier.predict(standard_scale)
    #print(input_predicted )

    if input_predicted == 0:
        st.title("The person is not diabetic")
    else:
        st.title("The person is diabetic")