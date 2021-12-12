import pandas as pd
import joblib as jb
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/tumors.csv')

X = df.iloc[:,0:2]
y = df.iloc[:,2]

reg = LinearRegression().fit(X, y)

jb.dump(reg, 'regression_trained.joblib')
