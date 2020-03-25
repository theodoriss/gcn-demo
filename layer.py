#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# In[ ]:


class GCN_layer(nn.Module):
    

    def __init__(self,inputs_shape,outputs_shape):
        super(GCN_layer, self).__init__()


        self.W=Parameter(torch.rand(inputs_shape,outputs_shape),requires_grad=True)
        self.bias = Parameter(torch.rand(outputs_shape),requires_grad=True)

    
    def forward(self,Adj_matrix,input_features):
        
        
        A=torch.from_numpy(Adj_matrix).type(torch.LongTensor)
        
        assert A.shape[0]==A.shape[1]
        I=torch.eye(A.shape[0])
        
        A_hat=A+I
        
        D=torch.sum(A_hat,axis=0)
        
        D=torch.diag(D)
        
        D_inv=torch.inverse(D)
                
        A_hat = torch.mm(torch.mm(D_inv,A_hat),D_inv)
        
        aggregate=torch.mm(A_hat,input_features)
        
        propagate=torch.mm(aggregate,self.W)+self.bias     
                
        
        return propagate

