import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('internpe\diabetes.csv')

x = data.drop('Outcome', axis = 1)
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

dt = DecisionTreeClassifier()
dt.fit(x_train_scaled, y_train)

"""diabetes_input = (5,166,72,19,175,25.8,0.587,51)
np_array_input = np.asarray(diabetes_input)
reshaped_input = np_array_input.reshape(1,-1)
scaled_input = sc.transform(reshaped_input)
prediction = dt.predict(scaled_input)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')"""

with open('diabetes_model.pkl', 'wb') as model_file:
  pickle.dump(dt, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
  pickle.dump(sc, scaler_file)