import p1
import p2
import p3
import p4
import pandas as pd
import matplotlib.pyplot as plt

website = str(input("Enter website name=> "))
p1.category1(website)
p2.category2(website)
p3.category3(website)
p4.category4(website)


read = pd.read_csv(r'phishing5.txt',header = None,sep = ',') #here we are reading a csv file with pandas module
read = read.iloc[:,:-1].values 
dataset = pd.read_csv(r'Training Dataset1.csv')
X = dataset.iloc[:,:-1].values 	#X independent variables 
y = dataset.iloc[:,-1].values      #Y is taken as the dependent variable in this case result 

from sklearn.model_selection import train_test_split  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1001)  # we have the test_size to be 20% that means only 20% of the dataset is the testing dataset 


from sklearn.ensemble import RandomForestRegressor  #regressor is used as here we are not taking the majority vote we are calculating the mean and std to predict the output
regressor = RandomForestRegressor(n_estimators = 10,criterion = "mse",random_state = 2)
regressor.fit(X_train,y_train)     #this will create the decision tree                        

y_pred = regressor.predict(X_test) #check the prediction with the test of independent variables


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = regressor,X=X_train,y=y_train,cv = 5) #a cross validation generator
accuracy.mean()  #calculates mean
accuracy.std()  #calculates standard deviation


Detect_phishing_website = regressor.predict(read)    #and then finally predicts the output by regressor with read data (the data from the files)

df=pd.read_csv(r'Training Dataset1.csv')  #bar chart of the result values which tells the number of legislative and phishing websites in the dataset
Y=df.iloc[:,-1].values 
arr_count=[0,0]
for value in Y:  
    if value==-1:
        arr_count[0]=arr_count[0]+1
    else:
        arr_count[1]=arr_count[1]+1
arr_values=['Legitimate','Phishing']  
plt.xlabel('Categories of Website',fontsize=18) 
plt.ylabel('count',fontsize=16)  
plt.bar(arr_values,arr_count,color=['red','yellow'])  
plt.show() #show the bar graph
print(arr_count)

if Detect_phishing_website == 1: 
    print("legitimate website")
elif Detect_phishing_website == 0:
    print ('suspicious website')
else:
    print('phishing website')
