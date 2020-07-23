# importing libraries for handling datasets
import pandas as pd
df = pd.read_csv("cardekho.csv")
# getting basic insights into data set
print(df.head())
print(df.tail())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.Fuel_Type.describe())
# Exploratory data analysis
##import pandas_profiling as pp
##profile = pp.ProfileReport(df)
##profile.to_file("CardekhoEDAreport.html")
# data wrangling
# removing duplicates
df.drop_duplicates(inplace=True)
print(df.shape)
#dropping unnecessary columns
df1=df.drop("Car_Name",axis=1)
print(df1.shape)
#treating cateogorical variables
df1=pd.get_dummies(df1,columns=["Fuel_Type","Seller_Type","Transmission"])
print(df1.Selling_Price.describe())
print(df1.head())
df1.Year= 2020-df1['Year']
print(df1.Year.describe())
#model building
y=df1["Selling_Price"]
X=df1.drop("Selling_Price",axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)
#model fitting
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred=regressor.predict(X_test)
print(y_pred)
print(y_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse=mean_squared_error(y_test,y_pred)
print(mse)
mae=mean_absolute_error(y_test,y_pred)
print(mae)
import math
rmse=math.sqrt(mse)
print(rmse)
print(df1.dtypes)
# Setting for deploymentmachine-learning-deployment-master
import pickle
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,12,3000,1,0,0,1,0,1,0,1]]))

