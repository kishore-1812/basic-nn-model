# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
From the given dataset, The model should be trained to give respective correct output for any given new input.

## Neural Network Model

![image](https://user-images.githubusercontent.com/63336975/187065284-9a7f7189-3c7a-49a4-acce-7ec8f0f9d45d.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

``` python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
data = pd.read_csv("/content/sample_data/random_dataset.csv")

df = pd.DataFrame(data = data)
df.head()
X = df[["input "]].values
Y = df[["output"]].values
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=50)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)
modified_train_x = Scaler.transform(X_train)
modified_test_x = Scaler.transform(X_test)
model = Sequential([
    Dense(9,activation='relu'),
    Dense(15,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(modified_train_x,y_train,epochs=5000)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
model.evaluate(modified_test_x,y_test)
pred1 = [[78]]
pred_trans = Scaler.transform(pred1)
model.predict(pred_trans)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/63336975/187065319-f96d083b-bb81-4dbb-8268-5326485f4218.png)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/63336975/187065347-1add0b3d-352f-4bdf-b04a-81a4e905b10f.png)

### Test Data Root Mean Squared Error

0.7378836274147034

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/63336975/187065399-d4bb9399-c25e-4d42-905a-0c902270f3c8.png)

## RESULT
Succesfully created and trained a neural network regression model for the given dataset.

