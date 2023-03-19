import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

crop_data=pd.read_csv("crop_production.csv")
crop_data.isnull().sum()
crop_data = crop_data.dropna()

#Label Encoding

encoder = LabelEncoder()
crop_data['State_Name'] = encoder.fit_transform(crop_data['State_Name'])

crop_data['Season'] = encoder.fit_transform(crop_data['Season'])

crop_data['Crop'] = encoder.fit_transform(crop_data['Crop'])


x = crop_data.drop(["Production","District_Name"], axis=1)
y = crop_data["Production"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)
model = RandomForestRegressor(n_estimators = 11)
model.fit(x_train,y_train)

pickle.dump(model, open('model1.pkl', 'wb'))