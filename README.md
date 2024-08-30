# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values. 
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.


```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
![307229421-e74202dc-3e2f-48db-8d2f-84dc9aa4cdeb](https://github.com/user-attachments/assets/f078b8d2-e868-4d15-aee8-637f592ef5ec)

![307229486-a9682a0c-6b81-46e3-b071-2e4a3aa50a1d](https://github.com/user-attachments/assets/bf1a29cf-67d0-4338-aa1d-b931ff98cf5d)

![307229553-7af1094c-e890-4a53-a684-00afb1f9b0f0](https://github.com/user-attachments/assets/9d45730f-f350-418c-a140-e57d291f1058)

![307229614-1c580e33-464f-47ad-9eff-4851bf17a91a](https://github.com/user-attachments/assets/119c03e7-d18e-4825-a8ce-76c4602c5426)

![307229663-a9411b7e-a555-48a3-952a-10b1e61726ae](https://github.com/user-attachments/assets/0f09e1d5-46c2-40af-8cec-0c9d163dd027)

![307229756-91beb994-6872-4bf0-8adb-0d31dcee50e1](https://github.com/user-attachments/assets/f5186416-2030-4477-9574-b5cfdb2f63ca)

![307229795-9d5fbbb4-70e3-4a8a-bffb-6b37bff77cbc](https://github.com/user-attachments/assets/c2e5a799-a0a7-4f64-b7f8-1034cf6548e4)

![307229843-a21e6f27-be9e-4879-9afa-b52b0290271c](https://github.com/user-attachments/assets/77b8b30c-f32f-447c-9eb1-93838b9136a2)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
