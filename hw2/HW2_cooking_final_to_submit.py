
# coding: utf-8

# In[40]:

import pandas as pd
import numpy as np
from sklearn import cross_validation
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from pylab import *
get_ipython().magic('matplotlib inline')

train=pd.read_json('train.json')


# In[7]:

train.shape


# In[8]:

train_np=np.array(train)


# In[9]:

#1b
def create_ingredient_list(dataset):
    ing_list=[]
    for i in range(0,len(dataset)):
        #make list of ingredients
        ings=train_np[i][2]
        for ing in ings:
            if ing in ing_list:
                continue
            else:
                ing_list.append(ing)
    return ing_list


# In[12]:

ing_list=create_ingredient_list(train_np)


# In[13]:

len(ing_list)


# In[10]:

#creates a list of the distinct cuisines
cuisine_list=[]
label_list=[]
for i in range(0,39774):
    cuisines=train_np[i][0]
    #create the y, i.e. label matrix for the overall matrix
    label_list.append(cuisines)
    for cuisine in cuisines:
        if cuisine in cuisine_list:
            continue
        else:
            cuisine_list.append(cuisine)


# In[11]:

len(cuisine_list)


# In[17]:

#turn label_list to a numpy array so we could comfortably implement cross validation later
label_list=np.array(label_list)


# In[27]:

#create a matrix of 39774 binary vectors, called ingredient_matrix
ingredient_matrix= np.empty([39774,6714])
#create binary vector, the length of ingredient list, and an empty ingredient matrix (6714x39774)
bin_vec=np.zeros(6714)
#iterate over the each recipe's ingredient list 
for i in range(0,39774):
    ings=train_np[i][2] 
    for ing in ings:
        #get the index of the ingredient from the list we composed, and +1 the binary vector at that index
        ing_ix=ing_list.index(ing)
        bin_vec[ing_ix]=1
    #add the binary vector of that recipe's ingredients to the matrix
    ingredient_matrix[i]=bin_vec
    bin_vec=np.zeros(6714)


# In[20]:

def getAccuracy(intake):
    # get accuracy of the KNN classifier in the cross-validation test
    train_i, test_i, classifier = intake
    X_train = ingredient_matrix[train_i]
    X_test = ingredient_matrix[test_i]
    y_train = label_list[train_i]
    y_test = label_list[test_i]
    classifier.fit(X_train, y_train)
    return classifier.score(X_test, y_test)


# In[24]:

from sklearn.naive_bayes import GaussianNB, BernoulliNB
def CrossValidation_3fold(size):
    accGaus=0
    accBern=0
    for train, test in (cross_validation.KFold(size, n_folds=3)):
        #question to self: implement Gaussian on original, non-binary dataset?
        gnb=GaussianNB()
        accGaus=float(getAccuracy([train,test,gnb])/3.0)+accGaus
        brn=BernoulliNB()
        accBern=float(getAccuracy([train,test,brn])/3.0)+accBern
    return accGaus, accBern


# In[28]:

gaus,ber=CrossValidation_3fold(39774)


# In[144]:

print ("the accuracy of Naive Bayer classifier assuming Gaussian distribution is " + str(gaus))


# In[145]:

print ("and assuming Bernoullian distribution, it is " + str(ber))


# In[15]:

from sklearn.linear_model import LogisticRegression
def logisticRegression_crossVal (size):
    accuracy=0
    for train, test in (cross_validation.KFold(size, n_folds=3)):
            lgr=LogisticRegression()
            accuracy=float(getAccuracy([train,test,lgr])/3.0)+accuracy
    return accuracy


# In[16]:

logReg=logisticRegression_crossVal(39774)


# In[143]:

print ("the accuracy of logistic regession classifier is " + str(logReg))


# In[33]:

test_data=pd.read_json('test.json')
test_np=np.array(test_data)


# In[41]:

from sklearn.linear_model import LogisticRegression


# In[42]:

#this function creates a binary matrix with as many rows as the dataset 
#and as many columns as the ingredient list (in this exercise, we will only use the ingredient list we trained on)
def create_binary_matrix(dataset,ing_list):
    ingredient_matrix= np.empty([len(dataset),len(ing_list)])
    bin_vec=np.zeros(len(ing_list))
    for i in range(0,len(dataset)):
        ings=dataset[i][1] 
        #iterate over the ingredient list of each recipe
        for ing in ings:
            #get the index of the ingredient from the list we composed, if it's in the ingredient list of our training array
            if ing in ing_list:
                ing_ix=ing_list.index(ing)
                bin_vec[ing_ix]=1
        ingredient_matrix[i]=bin_vec
        bin_vec=np.zeros(6714)
    return ingredient_matrix


# In[43]:

#creating the matrix of binary vectors (representing ingredients) to match that of our training datatset's
binary_test_matrix=create_binary_matrix(test_np, ing_list)


# In[44]:

#returns the predicted list of labels using the best classifier, training it on the binary training matrix 
#and running it on the binary test matrix we created
def kaggle(train_ingredient_list, labels_train, binary_test_matrix):
    classifier=LogisticRegression()
    classifier.fit(train_ingredient_list, labels_train)
    labels_test=classifier.predict(binary_test_matrix)
    return labels_test


# In[45]:

#stores the label list received from the kaggle function
result=kaggle(ingredient_matrix, label_list, binary_test_matrix)


# In[49]:

test_ids=[]
for i in range(0,len(test_np)):
    test_ids.append(test_np[i][0])


# In[50]:

#prints the dataframe to a csv file
cooking_output = pd.DataFrame({'id': test_ids, 'the_cuisine': result})
cooking_output.to_csv("cooking_submission.csv", index=False, index_label=False)
print (cooking_output)

