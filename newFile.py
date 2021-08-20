import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Training Dataset1.csv')
#dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

arr1=[]
arr2=[]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
arr = [.1,.2,.3,.4,.5, .60,.65,.70,.75,.80,.85,.90,.95]
for i in arr: 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
a = cm[0][0]
b=cm[0][1]
c=cm[1][1]
d=cm[1][0]
print('Random Forest->')
print('Total sum :',a+b+c+d)
print('sum of correct output :', a+c)
arr2.append(a+c)
print('Accuracy : ', (a+c)/(a+b+c+d))
arr1.append(((a+c)/(a+b+c+d))*100)
print('Confusion matrix: ')
print(cm)
# Fitting SVM to the Training set
classifier=SVC(kernel = 'rbf', random_state = 0) #classifier which uses SVC from sklearn.svm
classifier.fit(x_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred)
a = cm1[0][0]
b=cm1[0][1]
c=cm1[1][1]
d=cm1[1][0]
print('SVM -> ')
print('Total sum :',a+b+c+d)
print('sum of correct output :', a+c)
arr2.append(a+c)
print('Accuracy : ', (a+c)/(a+b+c+d))
arr1.append(((a+c)/(a+b+c+d))*100)
print('Confusion matrix: ')
print(cm1)


arr_values=['Random Forest','SVM']
plt.ylabel('Functionalities',fontsize=18)
plt.xlabel('Accuracy Percentage',fontsize=18)
plt.bar(arr_values,arr1,color=['orange','green'])
plt.show()

plt.ylabel('Functionalities',fontsize=18)
plt.xlabel('Sum of correct output',fontsize=18)
plt.bar(arr_values,arr2,color=['green','yellow'])
plt.show()

