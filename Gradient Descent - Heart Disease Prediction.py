#!/usr/bin/env python
# coding: utf-8

# In[60]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[61]:


#Reading the files
df = pd.read_csv("_cleveland-train.csv")


# In[62]:


#Check for null/NaN values
df.isnull().sum()


# In[63]:


#Converting all -1 to 0 for the cross-entropy function
df.loc[df["heartdisease::category|-1|1"] == -1, "heartdisease::category|-1|1"] = 0


# In[64]:


X = df.drop(['heartdisease::category|-1|1'], axis = 1)


# In[65]:


y = df['heartdisease::category|-1|1'].values


# In[66]:


print(y)


# In[67]:


#Getting the best features

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']
print(featureScores.nlargest(13,'Score'))


# In[68]:


#Not an important feature so we'll drop it.
X = df.drop(['fbs'],axis = 1)


# In[69]:


#Standardizing the features
sc = StandardScaler()
X = sc.fit_transform(X)


# In[70]:


#Getting the accuracy of the model:
def accuracy(y_pred,y_test):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count = count + 1
    return count / len(y_test)


# In[71]:


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)


# In[72]:


#Making transposes for later substitution
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T


# In[73]:


class myLogisticRegression:
    def __init__(self, rate = 0.00001, iters = 10000):
        self.rate = rate
        self.iters = iters
        self.W = None
        self.B = None
    
    #Initialize Parameters
    def initializeParas(self, features):
        self.W = np.zeros(shape = (features,1))
        self.B = 0
        return self.W, self.B
    
    #Sigmoid Function
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    
    #Gradient Descent
    def gradDFit(self,X,y):
        samples, features = X.shape
        losses = []
    
        #Initialize Parameters:
        W, B = self.initializeParas(samples)
    
    
        #Update values of weights and bias
        for i in range(self.iters):
            z = np.dot(self.W.T, X) + B
            y_pred = self.sigmoid(z)
        
        
            #Calculate cost
            loss = -(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
            cost = np.sum(loss) / X.shape[1]
            
        
    
            #Calculate gradients
            dw = (1 / X.shape[1]) * np.dot(X, (y_pred-y).T)
            db = (1 / X.shape[1]) * np.sum(y_pred - y)
        
        
            #Quit if below tolerance level
            if (dw < 0.001).all() and db < 0.001:
                break
                
        
        
            #Update the weight and bias - Gradient Descent
            self.W = self.W - self.rate * dw
            self.B = self.B - self.rate * db
            
        
        print(cost)
        return self.W, self.B, cost
    
    #Making the prediction function:
    def predict(self, X):
        features = X.shape[1]
        #y_pred = np.zeros((1,features))
        y_pred = []
        self.W = self.W.reshape(X.shape[0],1)
        z = np.dot(self.W.T, X) + self.B
        y_hat = self.sigmoid(z)
        
        #If sigmoid is >0.5, predict 1, otherwise predict -1
        for i in range(y_hat.shape[1]):
            if y_hat[:,i] > 0.5 :
                y_pred.append(1)
            else:
                y_pred.append(-1)
        
      
        return y_pred


# In[74]:


#Calling the class
model = myLogisticRegression(rate = 0.00001, iters=100000)


# In[75]:


#Fitting the model
project = model.gradDFit(X_train,y_train)


# In[76]:


#Predicting using the model's predict method
y_pred = model.predict(X_test)


# In[77]:


print(y_pred)


# In[78]:


#Convert -1 to 0 to check accuracy
for i in range(len(y_pred)):
    if y_pred[i] == -1:
        y_pred[i] = 0


# In[79]:


#Checking for the classification error
score = accuracy(y_test,y_pred)
classError = 1 - score
print(score)
print(classError)


# In[80]:


#Importing test data set
df2 = pd.read_csv("_cleveland-test.csv")


# In[81]:


X2 = sc.fit_transform(df2)


# In[82]:


print(X2)


# In[83]:


y_prediction = model.predict(X2.T)


# In[84]:


print(y_prediction)


# In[85]:


#Converting to array for writing to file purposes
Y_Pred = np.asarray(y_prediction)


# In[86]:


#Write the prediction file to another file
Y_Pred.tofile('HD5.csv', sep = '\n')


# In[87]:


#Checking sklearn's model's accuracy
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter = 100000)
lr.fit(X_train.T, y_train.T)


# In[88]:


Y_PRED = lr.predict(X_test.T)


# In[89]:


print(Y_PRED)


# In[90]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_PRED)
print(cm)
accuracy_score(y_test,Y_PRED)


# In[ ]:




