# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: V.Shriyha
RegisterNumber: 212224230267
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
# Data Head:
![Screenshot 2025-04-23 093138](https://github.com/user-attachments/assets/d28e6b62-7977-4e58-9c58-58354523fca8)


# Data Info:
![Screenshot 2025-04-23 093128](https://github.com/user-attachments/assets/507a8735-f043-4c86-bcab-28c5bc8dcb71)


# isnull() sum():
![Screenshot 2025-04-23 093117](https://github.com/user-attachments/assets/a3551a84-fe3e-4f88-9bc2-c3827cd51ea3)


# Data Head for salary:
![Screenshot 2025-04-23 093109](https://github.com/user-attachments/assets/73290d71-78a8-40e7-86a8-3d23f51e5440)


# Mean Squared Error :
![Screenshot 2025-04-23 093103](https://github.com/user-attachments/assets/ebac183f-699d-43ca-96fb-dcad1ae23d76)


# r2 Value:
![Screenshot 2025-04-23 093054](https://github.com/user-attachments/assets/6aecd1dc-3554-4ec2-94d0-f3aab7540807)


# Data prediction :
![Screenshot 2025-04-23 093050](https://github.com/user-attachments/assets/d61b03f4-cd43-41b5-8df6-4748becde16c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
