import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
df = pd.read_csv('E:\Data_Science\Logistic Regression\logistic.csv')
print(df)
x_train,x_test,y_train,y_test = train_test_split(df[['age']],df['have_insurance'],test_size = 0.3)
print(len(x_test))
print(len(x_train))
model = LogisticRegression()
model1 = LinearRegression()
model1.fit(x_train,y_train)
model.fit(x_train,y_train)
cm = confusion_matrix(y_test,model.predict(x_test))
print(x_test)
print(model.predict(x_test))
plt.xlabel('Ages')
plt.ylabel('Have_insurence')
plt.title('Scatter Plot Of Age V/S Insurence')
plt.scatter(df.age,df.have_insurance,marker = "+",color = 'r')
plt.plot(x_test,model1.predict(x_test),color = 'b')
sn.heatmap(cm)
plt.show()
