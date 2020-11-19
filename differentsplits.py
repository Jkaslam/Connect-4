import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

#data = pd.read_csv("connect-4data.csv")

Wins = pd.read_csv("Wins.csv")
Losses = pd.read_csv("Losses.csv")
Draws = pd.read_csv("Draws.csv")

numw=list(range(0,np.size(Wins,0)-1))
numl=list(range(0,np.size(Losses,0)-1))
numd=list(range(0,np.size(Draws,0)-1))


# Make the data numerical by turning b -> 1, x-> 2, o -> 3
#data.replace(to_replace =['b', 'x', 'o', 'win', 'loss', 'draw'], value=[1, 2, 3, 1, 2, 3], inplace=True)
Wins.replace(to_replace =['b', 'x', 'o', 'win', 'loss', 'draw'], value=[1, 2, 3, 1, 2, 3], inplace=True)
Losses.replace(to_replace =['b', 'x', 'o', 'win', 'loss', 'draw'], value=[1, 2, 3, 1, 2, 3], inplace=True)
Draws.replace(to_replace =['b', 'x', 'o', 'win', 'loss', 'draw'], value=[1, 2, 3, 1, 2, 3], inplace=True)

#features = data.iloc[:, :-1]

# Hot encode the labels, i.e. convert the integers representing classes to binary
hot_encoder = OneHotEncoder();
#labels = hot_encoder.fit_transform(data.iloc[:, [-1]]).toarray()


props=np.array([[6449, 6449, 6449],[16635, 16635, 6449],[25000, 16635, 6449],[44473, 16635, 6449]])



wins_accuracy=[]
losses_accuracy=[]
draws_accuracy=[]

wins_count=[]
losses_count=[]
draws_count=[]

for k in range(4):
    
    
    testwins = props[k,0]
    testlosses = props[k,1]
    testdraws = props[k,2]
    
    wind=random.choices(numw,k = testwins)
    lind=random.choices(numl,k = testlosses)
    dind=random.choices(numd,k = testdraws)
    
    D = Wins.iloc[wind,:]
    D=D.append(Losses.iloc[lind,:])
    D=D.append(Draws.iloc[dind,:])
    data = D.iloc[:,1:44]
    
    features = data.iloc[:, :-1]
    

    labels = hot_encoder.fit_transform(data.iloc[:, [-1]]).toarray()
    
    model_accuracies = list()
    #for j in range(1):
        # Split data into training data and testing data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = .7)

        # Configure the neural network
    model = Sequential()
    model.add(Dense(40, input_dim=42, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the neural network on the input data
    history = model.fit(train_features.to_numpy(), train_labels, epochs=100, batch_size = 64, verbose=0)

        # Figures out the accuracy of the current model
    pred_softmax = model.predict(test_features)
    pred_labels = list()
    for i in range(len(pred_softmax)):
        pred_labels.append(np.argmax(pred_softmax[i]))
    unencoded_test_labels = list()
    for i in range(len(test_labels)):
        unencoded_test_labels.append(np.argmax(test_labels[i]))
    
    test_accuracy = accuracy_score(pred_labels, unencoded_test_labels)
        #print(test_accuracy)
    model_accuracies.append(test_accuracy)

    
    
#print("The total test accuracy is:", test_accuracy)
    test_features['label'] = unencoded_test_labels

# Separate the win test data and make predictions on it
    test_wins = test_features[test_features['label']== 0]
    win_pred_soft_max = model.predict(test_wins.iloc[:,:-1])

    win_predictions = list()
    for i in range(len(win_pred_soft_max)):
        win_predictions.append(np.argmax(win_pred_soft_max[i]))
        
    wins_accuracy.append(accuracy_score(win_predictions, [0]*len(win_predictions)))
    wins_count.append(len(test_wins))
    
# Separate the loss test data and make predictions on it
    test_losses = test_features[test_features['label']== 1]
    loss_pred_soft_max = model.predict(test_losses.iloc[:,:-1])

    loss_predictions = list()
    for i in range(len(loss_pred_soft_max)):
        loss_predictions.append(np.argmax(loss_pred_soft_max[i]))
        
    losses_accuracy.append(accuracy_score(loss_predictions, [0]*len(loss_predictions)))
    losses_count.append(len(test_losses))
    
# Separate the draw test data and make predictions on it
    test_draws = test_features[test_features['label']== 2]
    draw_pred_soft_max = model.predict(test_draws.iloc[:,:-1])

    draw_predictions = list()
    for i in range(len(draw_pred_soft_max)):
        draw_predictions.append(np.argmax(draw_pred_soft_max[i]))
        
    draws_accuracy.append(accuracy_score(draw_predictions, [0]*len(draw_predictions)))
    draws_count.append(len(test_draws))
    

print("The number of wins in the test set is:", wins_count)
print("The accuracy of the model on wins is", wins_accuracy)
print("The number of losses in the test set is:", losses_count)
print("The accuracy of the model on losses is", losses_accuracy)
print("The number of draws in the test set is:", draws_count)
print("The accuracy of the model on draws is", draws_accuracy)
