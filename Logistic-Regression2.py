import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("E:\Data_Science\Logistic Regression\logistic.csv")
print(data)
plt.xlabel(" Ages")
plt.ylabel(" Have Insurence ")
plt.scatter(data.age,data.have_insurance)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(data[['age']],data.have_insurance,test_size = 0.2)
print(len(x_train))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
print(x_test)
print(model.predict((x_test)))
print(model.predict_proba(x_test))
print(model.score(x_test,y_test))
y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
print(cm)
import seaborn as sn
sn.heatmap(cm,annot=True)
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()


