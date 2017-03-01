
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[80]:

import pandas as pd
import numpy as np
from numpy import linalg
import re
from sklearn import preprocessing
from nltk.stem.porter import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import k_means
from sklearn.metrics import accuracy_score


# #### Load the Data

# In[81]:

def load(file):
    df = []
    with open(file, 'r') as f:
        for line in f.readlines():
            n = line.strip().split('\t')
            if len(n) == 2:
                df.append({'review' : n[0], 'label' : int(n[1])}) 
    return pd.DataFrame(df)


# In[82]:

#Load the Data
yelp = load('yelp_labelled.txt')
amazon = load('amazon_cells_labelled.txt')
imdb = load('imdb_labelled.txt')


# #### Inspect Ratio

# In[83]:

yelp['label'].value_counts()


# In[84]:

amazon['label'].value_counts()


# In[85]:

imdb['label'].value_counts()


# #### Pre-Processing

# In[86]:

#Lowercase:
yelp['review'] = yelp['review'].str.lower()
amazon['review'] = amazon['review'].str.lower()
imdb['review'] = imdb['review'].str.lower()

yelp.dropna()
amazon.dropna()
imdb.dropna()


# In[87]:

punctuation = re.compile(r'([^A-Za-z0-9 ])')
def puncRem(review):
    return punctuation.sub("", review)


# In[88]:

#http://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html
stop_words = (['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
st = PorterStemmer()
def StemStop(review):
    return ' '.join([st.stem(word) for word in review.split() if word not in stop_words])


# In[89]:

def cleanAll(file):
    rev = list(file['review'])
    def clean(revs):
        return StemStop(puncRem(revs))
    return [clean(row) for row in rev]


# In[90]:

#Cleaned Files
amazonCl = amazon
yelpCl = yelp
imdbCl = imdb

amazonCl['review'] = cleanAll(amazon)
yelpCl['review'] = cleanAll(yelp)
imdbCl['review'] = cleanAll(imdb)


# ### Split Train/Test

# In[91]:

def train(file):
    label0,label1=file[file.label==0],file[file.label==1]
    train0,test0=label0[:400],label0[400:]
    train1,test1 = label1[:400],label1[400:]
    return train1.append(train0, ignore_index = True), test1.append(test0, ignore_index = True)


# In[92]:

imdbTrain, imdbTest = train(imdbCl)
yelpTrain, yelpTest = train(yelpCl)
amazonTrain, amazonTest = train(amazonCl)


# In[93]:

trainSet=imdbTrain.append(amazonTrain,ignore_index=True).append(yelpTrain, ignore_index=True)
testSet=imdbTest.append(amazonTest,ignore_index=True).append(yelpTest, ignore_index=True)


# ### Bag of Words

# In[94]:

wDict={}
wList=[]


# In[95]:

def wordLoc(file):
    p = 0
    for review in file['review']:
        for word in review.split(' '):
            if not word in wDict:
                wDict[word] = p
                p += 1
                wList.append(word)


# In[96]:

wordLoc(trainSet)


# In[97]:

def featureExt(file):
    x=len(wDict)
    y=len(file['review'])
    num=np.zeros((y,x),dtype=np.int)
    for p,review in enumerate(file['review']):
        for word in review.split(' '):
            if word in wDict:
                num[p][wDict[word]]+=1
    return num


# In[98]:

def cen(file):
    avg=np.mean(file,axis=0)
    return file-avg


# In[99]:

featureTrain = featureExt(trainSet)
featureTest = featureExt(testSet)


# In[100]:

trainL=trainSet['label']
testL=testSet['label']


# In[101]:

import matplotlib.pyplot as plt
plt.xlabel('Label')
plt.ylabel('Number of Occurences in Feature Vector')
plt.hist(featureTrain[0])


# In[102]:

plt.xlabel('Label')
plt.ylabel('Number of Occurences in Feature Vector')
plt.hist(featureTrain[1])


# ### E -- PreProcess

# In[103]:

def l2Norm(file,norm='l2'):
    return preprocessing.normalize(file.astype(np.float64), norm=norm)


# In[104]:

nFeatTrain = l2Norm(featureTrain)


# In[105]:

nFeatTest = l2Norm(featureTest)


# #### F -- Clustering

# In[106]:

#calc distance
def l2D(a,b):
    return np.linalg.norm(a-b)


# In[107]:

def centPoint(cents,points):
    r=[None for _ in range(len(cents))]
    for point in points:
        i=centers(cents,point)
        if r[i] is None:
            r[i]=point
        else:
            r[i]=np.vstack((r[i],point))
    return r


# In[108]:

def centers(cents,a):
    resI,dist=-1,float('inf')
    for i,cent in enumerate(cents):
        newD=l2D(cent,a)
        if newD < dist:
            resI,dist = i,newD
    return resI


# In[109]:

#K-Centroids
def kCen(s,feat,K=2):
    def end(c1,c2):
        for i in range(len(c1)):
            cIn, next = c1[i],c2[i]
            if not np.allclose(cIn,next,rtol=1e-4,atol=1e-6):
                return False
        return True
    m=centPoint(s,feat)
    c1, c2 = s,[np.mean(mat,axis=0) for mat in m]
    t = 1
    while not end(c1,c2):
        print("rep " + str(t))
        t += 1
        m=centPoint(c2,feat)
        c1 = c2
        c2 = [mat.mean(0) for mat in m]
    l=np.zeros((feat.shape[0],1), dtype=np.int)
    for index, point in enumerate(feat):
        i=centers(c2, point)
        l[index, 0] = i
    return c2,l


# In[110]:

#p1,p2=np.random.choice(nFeatTrain.shape[0],2,replace=False)
#s=nFeatTrain[[p1,p2]]
#cen,lab=kCen(s,nFeatTrain)


# In[111]:

print(cen[0])


# In[112]:

print(cen[1])


# In[113]:

accuracy_score(trainL,lab)


# #### G Sentiment Prediction

# In[114]:

logreg = LogisticRegression()
logreg.fit(nFeatTrain,trainL)


# In[115]:

classification_accuracy=logreg.score(nFeatTest, testL)


# In[116]:

classification_accuracy


# In[117]:

pred=logreg.predict(nFeatTest)
rank=logreg.coef_


# In[118]:

cm=confusion_matrix(testL,pred)


# In[119]:

cm


# In[ ]:

print('Top Negative Words:')
for negativeWords in rank.argsort()[0,:10]:
    print(wList[negativeWords])


# In[120]:

print('Top Positive Words:')
for positiveWords in rank.argsort()[0,-10:]:
    print(wList[positiveWords])


# #### H -- N-Gram Model 

# In[122]:

nD={}
nL=[]


# In[204]:

def nGram(file,n=2):
    y=len(file['review'])
    x=len(nD)
    df=np.zeros((y,x),dtype=np.int)
    for i,review in enumerate(file['review']):
        sp=review.split(' ')
        for spI in range(len(sp)-n+1):
            gr=' '.join(sp[spI: (spI+n)])
            if gr in nD:
                df[i][nD[gr]] += 1
    return df


# In[205]:

nGrTrain=nGram(trainSet)


# In[224]:

nGrTrain=nGram(trainSet)


# In[225]:

nGrTest=nGram(testSet)


# In[226]:

#clustering


# In[227]:

normNTrain = l2Norm(nGrTrain)


# In[228]:

normNTest = l2Norm(nGrTest)


# ### N-Gram Clustering

# In[238]:

#p1,p2=np.random.choice(normNTrain.shape[0],2,replace=False)
#sN=nFeatTrain[[p1,p2]]
#cenN,labN=kCen(sN,normNTrain)


# In[240]:

#accuracy_score(trainL,labN)


# ##### N-Gram LogReg

# In[242]:

logreg=LogisticRegression()


# In[243]:

logreg.fit(nGrTrain,trainL)


# In[244]:

logreg.score(nGrTest,testL)


# In[245]:

pred=logreg.predict(normNTest)


# In[246]:

rank=logreg.coef_


# In[247]:

cm=confusion_matrix(testL,pred)


# In[248]:

cm


# In[ ]:

print('Top Negative Words:')
for negativeNWords in rank.argsort()[0,:10]:
    print(nL[negativeNWords])


# In[249]:

print('Top Positive Words:')
for positiveNWords in rank.argsort()[0,-10:]:
    print(nL[positiveNWords])


# ### PCA 

# In[251]:

def pcaBow(file,n=10):
    U,s,V=np.linalg.svd(file,full_matrices=True)
    print(U.shape)
    diag=np.diag(s[:n])
    print(diag.shape)
    return np.dot(U[:,:n], diag)


# In[262]:

def orig(f,n=2):
    p1,p2=np.random.choice(f.shape[0],2, replace=False)
    st=f[[p1,p2]]
    return st


# In[274]:

#10 Dim
pca10=pcaBow(featureTrain,10)


# In[275]:

init10=orig(pca10)


# In[276]:

cenP10,labP10=kCen(init10,pca10)


# In[277]:

print(cenP10)


# In[278]:

accuracy_score(trainL, labP10)


# In[ ]:

#50 Dim


# In[285]:

pca50=pcaBow(featureTrain,50)


# In[286]:

init50=orig(pca50)


# In[287]:

cenP50,labP50=kCen(init50,pca50)


# In[288]:

print(cenP50)


# In[289]:

accuracy_score(trainL,labP50)


# In[290]:

#100 Dim


# In[291]:

pca100=pcaBow(featureTrain,100)


# In[292]:

init100=orig(pca100)


# In[293]:

cenP100,labP100=kCen(init100,pca100)


# In[294]:

print(cenP100)


# In[295]:

accuracy_score(trainL,labP100)

