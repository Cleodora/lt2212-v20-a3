import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from utils import randnumber, separateindexes
from random import randrange
from sklearn.metrics import classification_report
import config as conf
# Whatever other imports you need

# You can implement classes and helper functions here too.
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        if self.hidden_size > 0:
            self.fc1 = nn.Linear(self.input_size * 2, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, 1) 
        else:
            self.fc1 = nn.Linear(self.input_size * 2, 1)
        if self.activation:
            self.nonlinearity = conf.activation[self.activation]
        self.sigmoid = nn.Sigmoid() 

    def forward(self, X):
        X = self.fc1(X)
        if self.activation and self.hidden_size > 0:
            X = self.nonlinearity(X)
        if self.hidden_size > 0:
            X = self.fc2(X)
        X = self.sigmoid(X)
        return X




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hidden", dest="hiddensize", type=int, default="0", help="The size of the hidden Layer")
    parser.add_argument("--activation", dest="activation", type=str, default=None, help="Choices are relu, elu or tanh", choices= ['relu', 'elu', 'tanh', None])
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    df = pd.read_csv(args.featurefile)
    train_X = df.loc[df['train_test_tag'] == 0]
    test_X = df.loc[df['train_test_tag'] == 1]
    train_X = train_X.iloc[:, 1:]
    test_X = test_X.iloc[:, 1:]
    test_X = test_X.reset_index()
    separate_indexes_labels_train = train_X['labels']
    separate_indexes_labels_test = test_X['labels']
    train_X = train_X.drop(['labels','train_test_tag','original_author_name'], axis = 1)
    test_X = test_X.drop(['labels','train_test_tag','original_author_name', 'index'], axis = 1)

    
    separate_indexes_labels_train, unique_values, d = separateindexes(separate_indexes_labels_train)
    separate_indexes_labels_test, unique_values1, d1 = separateindexes(separate_indexes_labels_test)
    
    input_size = len(train_X.columns)
    model = FFNN(input_size, args.hiddensize, args.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr = conf.parameters['LEARNING_RATE'])
    criterion = nn.BCELoss()
    Y = []
    X1 = []
    for e in range(conf.parameters['EPOCHS']):
        print("EPOCH", e)
        for k,v in d.items():
            total_loss = 0
            current_author_indexes = v
            amount_current_author_indexes = len(current_author_indexes)
            other_than_classes = randnumber(current_author_indexes, len(separate_indexes_labels_train))
            for ins, en in enumerate(v):
                total_loss = 0
                x1 = torch.FloatTensor(np.array(train_X.iloc[en]))
                x2 = []
                if randrange(2) == 0:
                    rand2 = randrange(len(other_than_classes))
                    x2 = torch.FloatTensor(np.array(train_X.iloc[other_than_classes[rand2]]))
                    other_than_classes.remove(other_than_classes[rand2])
                    Y.append(0)
                else:
                    if ins < amount_current_author_indexes - 1:
                        x2 = torch.FloatTensor(np.array(train_X.iloc[v[ins + 1]]))
                    else:
                        continue
                    Y.append(1)
                x = torch.cat((x1, x2))
                optimizer.zero_grad() 
                outputs = model(x)
                X1.append(outputs)
                # print(outputs)

            X2 = torch.cat(X1, dim = 0)    
            # print(X2)
            ro = [torch.FloatTensor([pre]) for pre in Y]
            ro = torch.FloatTensor(ro)
            X1 = [torch.FloatTensor([pre]) for pre in Y]
            # ro = torch.FloatTensor(ro)
            loss = criterion(X2, ro)
            total_loss += loss.item()
            # print("Total Loss", total_loss)
            loss.backward()
            optimizer.step()

    Y1 = []
    # X1 = [] 
    pred = []
    pred2 = []
    counter = 0
    with torch.no_grad():
        for k,v in d1.items():
            current_author_indexes = v
            amount_current_author_indexes = len(current_author_indexes)
            other_than_classes = randnumber(current_author_indexes, len(separate_indexes_labels_test))
            for ins, en in enumerate(v):
                x1 = torch.FloatTensor(np.array(test_X.iloc[en]))
                x2 = []
                if randrange(2) == 0:
                    rand2 = randrange(len(other_than_classes))
                    x2 = torch.FloatTensor(np.array(train_X.iloc[other_than_classes[rand2]]))
                    other_than_classes.remove(other_than_classes[rand2])
                    Y1.append(0)
                else:
                    if ins < amount_current_author_indexes - 1:
                        x2 = torch.FloatTensor(np.array(train_X.iloc[v[ins + 1]]))
                    else:
                        continue
                    Y1.append(1)
                x = torch.cat((x1, x2))
                outputs = model(x)
                pred2.append(outputs)
                prediction = 1 if outputs > 0.5 else 0
                pred.append(prediction)

    print(classification_report(Y1, pred))



    # for epoch in epochs:


    
    # implement everything you need here
    
