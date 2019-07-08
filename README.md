# LinearRegression
import numpy as np import
pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
#%matplotlib inline 
DataFrame=pd.read_csv('C:\Users\DILIP\Desktop\pandas\dilip\Salary.csv')  
X=DataFrame.iloc[:,:-1].values 
y=DataFrame.iloc[:,1].values  
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)  
from sklearn.linear_model import LinearRegression 
SimpleLinearRegression= LinearRegression() 
SimpleLinearRegression.fit(X_train,y_train) 
y_predict=SimpleLinearRegression.predict(X_test) 
#print(y_predict)
#print(SimpleLinearRegression.coef_) 
#print(SimpleLinearRegression.intercept_)  
plt.scatter(X_train,y_train)
plt.plot(X_train,SimpleLinearRegression.predict(X_train)) 
plt.show()  
from sklearn.metrics import r2_score r2_score(y_test,y_predict) 
sns.heatmap(DataFrame.corr())

