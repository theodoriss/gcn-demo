#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F

import train
import utils
import layer


# In[2]:


graph=utils.create_Graphs_with_attributes('karate.edgelist.txt','karate.attributes.csv')


# In[3]:


A = np.array(nx.to_numpy_matrix(graph)) # adjadjency matrix


# In[4]:


X_train,Y_train,X_test,Y_test=utils.create_train_test(graph)


# In[5]:


class GCN(nn.Module):
    

    def __init__(self,inputs_shape,outputs_shape,n_classes,activation='Relu'):
        super(GCN, self).__init__()

        self.layer1=layer.GCN_layer(inputs_shape,outputs_shape)
        self.layer2=layer.GCN_layer(outputs_shape,n_classes)
        
        
        if activation =='Tanh':
            self.activation = nn.Tanh()
        elif activation=='Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation=='Softmax':
            self.activation=nn.Softmax()
        elif activation=='Relu':
            self.activation=nn.ReLU()
    
        self.softmax=nn.Softmax()
        
    
    def forward(self,Adj_matrix,input_features):
        
        x=self.layer1(Adj_matrix,input_features)
        x=self.activation(x) 
        x=self.layer2(Adj_matrix,x)
        x=self.softmax(x)     
        
        return x


# In[6]:


model=GCN(inputs_shape=utils.create_features(graph).shape[1],outputs_shape=4,n_classes=2,activation='Tanh')


# In[7]:


trainer = train.Trainer(
    model,
    optimizer = optim.Adam(model.parameters(), lr=0.01),
    loss_function=F.cross_entropy,
    epochs=250
)


# In[8]:


trainer.train(X_train,Y_train)


# In[9]:


trainer.test(X_test,Y_test)


# In[10]:


trainer.visualize_classification(graph,Y_test,classification=True)


# In[ ]:





# In[ ]:




