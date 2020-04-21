#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using Iris dataset to compare the performance of the various machine learning algos


# In[ ]:


#importing packages


# In[66]:


import pandas as pd # linear algebra
import numpy as np # data processing
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#read the iris dataset from the desktop


# In[4]:


iris = pd.read_csv('iris.csv')


# In[37]:


# Get a glimpse of columns of iris dataset


# In[5]:


iris.columns


# In[7]:


iris.head(2) # show first two rows of the dataset


# In[10]:


iris = iris.drop(columns ='Id') # ID is not needed so dropping it


# In[11]:


iris.head(2) 


# In[12]:


#Data Visualization with Iris dataset


# In[13]:


iris.hist(edgecolor='black')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


#Observation: SepalWidthcm is nearly normally distributed as compared to the other covariates


# In[14]:


# Find the correlation between the variables
plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# In[ ]:


#Obervation:There is high correlation observed in the PetalLengthCm and PetalWidthCm with the SepalLengthCm. 
#Also, high correlation is observed between PetalWidthCm and PetalLengthCm.SepalLength and SepalWidth are not correlated.


# In[15]:


help(corr)


# In[21]:


scatter1 = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='pink', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='yellow', label='versicolor',ax=scatter1)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=scatter1)
scatter1.set_xlabel("Sepal Length")
scatter1.set_ylabel("Sepal Width")
scatter1.set_title("Sepal Length vs Sepal Width")
scatter1=plt.gcf()
scatter1.set_size_inches(12,8)
plt.show()


# In[ ]:


#Observation: There is overlap of datapoints between the versicolor and virginica species 


# In[23]:


scatter1 = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='pink', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='yellow', label='versicolor',ax=scatter1)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=scatter1)
scatter1.set_xlabel("Petal Length")
scatter1.set_ylabel("Petal Width")
scatter1.set_title("Petal Length vs Petal Width")
scatter1=plt.gcf()
scatter1.set_size_inches(12,8)
plt.show()


# In[ ]:


#Observation: All the three species look well splitted. This can be a good place to apply for the classification.


# In[ ]:


#Encoding for the target variable


# In[31]:



from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[ ]:


# imported all the libraries needed for the machine learning


# In[33]:


train, test = train_test_split(iris, test_size = 0.3) # It will divide the dataset into train (70%) and test (30%)


# In[34]:


train.shape


# In[35]:


test.shape


# In[39]:


train_x = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #taking the training data variables
train_y=train.Species  #output value of train data
test_x= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #taking test data variables
test_y =test.Species   #output value of test data


# In[ ]:


# Logistic regression


# In[40]:


model = LogisticRegression()#select the algorithm
model.fit(train_x,train_y)  #we train the algorithm with the training data and the training output
prediction=model.predict(test_) #now we pass the testing data to the trained algorithm
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))


# In[ ]:


#Observation: The accuracy of the algorithm is .98, which is quite high


# In[41]:


#Support Vector Machines


# In[43]:


model = svm.SVC() #select the algorithm
model.fit(train_x,train_y) # we train the algorithm with the training data and the training output
prediction=model.predict(test_x) #now we pass the testing data to the trained algorithm
print('The accuracy of the model is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output 


# In[ ]:





# In[ ]:


# The accuracy through SVM is .96, which is slightly lower than the accuracy through the Logistic model


# In[ ]:


#K-NN


# In[44]:


model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))


# In[ ]:


#The accuracy through the k-nn model is .933


# In[ ]:


#Decision Trees


# In[45]:


model=DecisionTreeClassifier()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))


# In[ ]:


#The accuracy through the Decision Tree model is .93


# In[ ]:





# In[ ]:


# Creating Sepal and Petal Training Data


# In[49]:





# In[48]:


petal=iris[['PetalLengthCm','PetalWidthCm','Species']] # defining the petal feature variables
sepal=iris[['SepalLengthCm','SepalWidthCm','Species']] # defining the sepal feature variables


# In[ ]:


# Petal variable set


# In[50]:


train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)
train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]
train_y_p=train_p.Species
test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]
test_y_p=test_p.Species


# In[ ]:


# Sepal variable set


# In[51]:


train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal
train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]
train_y_s=train_s.Species
test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]
test_y_s=test_s.Species


# In[ ]:


#Running Logistic regression for Sepal and Petal


# In[65]:


model = LogisticRegression()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# In[ ]:





# In[53]:


#The accuracy of the Logistic Regression using Petals is: 0.69
#The accuracy of the Logistic Regression using Sepals is: 0.64


# In[ ]:


# Running the SVM ML algorithm


# In[62]:


model=svm.SVC()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model=svm.SVC()
model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))


# In[ ]:





# In[ ]:


#The accuracy of the SVM using Petals is: 0.97
#The accuracy of the SVM using Sepal is: 0.8


# In[55]:


#Decision Tree


# In[63]:


model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# In[ ]:





# In[ ]:


#The accuracy of the Decision Tree using Petals is: 0.96
#The accuracy of the Decision Tree using Sepals is: 0.64


# In[ ]:


#Knn


# In[64]:


model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# In[ ]:


#The accuracy of the Decision Tree using Petals is: 0.97
#The accuracy of the Decision Tree using Sepals is: 0.73


# In[ ]:





# In[ ]:


#As we can see from the results after running allthe models, the petals dataset have higher accuracy as compared to sepals.


# In[ ]:


#This confirms with the correlation matrix which showed that petalWidth and petalLength has high correlation

