#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from sklearn.metrics import classification_report
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networkx as nx


import utils


# In[ ]:


graph=utils.create_Graphs_with_attributes('karate.edgelist.txt','karate.attributes.csv')

A = np.array(nx.to_numpy_matrix(graph)) # adjadjency matrix


# In[ ]:


class Trainer():
    def __init__(self,model,optimizer,loss_function,epochs):
        
        self.model=model
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.epochs=epochs
        
    def train(self,X_train,Y_train):



        y_train=torch.from_numpy(Y_train.astype(int)).type(torch.LongTensor)


        tot_loss=0.0

        all_preds=[]

        for t in range(self.epochs):

            epoch_loss = 0.0

            #model.train()

            y_pred=self.model(A,utils.create_features(graph))

            all_preds.append(y_pred)

            loss = self.loss_function(y_pred[X_train],y_train)

            self.optimizer.zero_grad()

            epoch_loss+=loss
            tot_loss+=loss


            loss.backward()
            self.optimizer.step()

            print(str(t),'epoch_loss:'+str(epoch_loss),'total loss:'+str(tot_loss))

        self.all_preds=all_preds
    
    def test(self,X_test,Y_test):

        self.model.eval()

        y_test=torch.from_numpy(Y_test.astype(int)).type(torch.LongTensor)

        y_pred=self.all_preds[-1]  # preds of last epoch


        loss_test = self.loss_function(y_pred[X_test],y_test)


        print('validation loss is equal to: '+str(loss_test))
    
    def visualize_classification(self,graph,Y_test,classification):
        last_epoch = self.all_preds[self.epochs-1].detach().numpy() # get outputs of last epoch
        predicted_class = np.argmax(last_epoch, axis=-1) # take the unit with the higher probability
        color = np.where(predicted_class==0, 'c', 'r')
        pos = nx.kamada_kawai_layout(graph)
        nx.draw_networkx(graph, pos, node_color=color, with_labels=True, node_size=300)
        if classification==True:
            print(classification_report(predicted_class[1:-1],Y_test))



