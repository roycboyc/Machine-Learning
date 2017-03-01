
# coding: utf-8

# In[46]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


# In[47]:

#define the input
n=np.linspace(0,10,100)


# In[48]:

#assign weights
weight=np.array([2,float(5)/3,float(5)/3,float(11)/3,float(5)/3])
weight1=np.array([1,-1,1,-1,1])


# In[60]:

#assign bias
bias=np.array([-2,float(-10)/3,float(-25)/3,-22,-15])
bias1=0


# In[61]:

#max f(x)
def nnFA(n):
    return np.maximum(n,0)


# In[62]:

vals=[]


# In[63]:

for i in range(len(n)):
    fin=np.array(n[i]).dot(weight)+bias
    sec=nnFA(fin)
    sec2=sec.dot(weight1)+bias1
    vals.append(sec2)


# In[75]:

range(0,10)


# In[77]:

plt.title('"Pride Rock"')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.plot(n,vals)
plt.axis([0, 10, -1, 6])
plt.xticks(range(0,11))
plt.show()

