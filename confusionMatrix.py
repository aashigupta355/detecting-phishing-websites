import ast
import dphishing
import warnings
import matplotlib.pyplot as plt

with open('phishing5.txt', 'r') as f:
    mylist = ast.literal_eval(f.read())

# confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
with open('phishing5.txt', 'r') as f:
    mylist = ast.literal_eval(f.read())
actual = [-1,0,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1] 

predicted = mylist
results = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score : ',accuracy_score(actual, predicted))

#plotting of the pie chart between correctly predicted and incorrectly predicted
value=0
for i in range (0,3):
    for j in range (0,3):
        if(i==j):
            value=value+results[i][j]   
x=[]
x.append(17-value)
x.append(17)
y=['Incorrectly Predicted','Correctly Predicted']  
plt.pie(x,labels=y,radius=1.2,shadow=True)  
plt.show()  


    
