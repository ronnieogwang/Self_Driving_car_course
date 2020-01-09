import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  #use datasets from sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

n_pts = 500
#factor show relative size of inner and outter circles, diameter of inner cirlce to outer circle
X, y = datasets.make_circles(n_samples = n_pts, random_state = 123, noise = 0.1, factor = 0.2)#noise creates Standard deviation in the points

#plot
plt.scatter(X[y==0, 0], X[y==0, 1] )
plt.scatter(X[y==1, 0], X[y==1, 1] )

#this data cannot be linaerly separable, thus we use a deep NN
classifier = Sequential()

#define hidden layer and specify input shape
#4 nodes, we need about four lines to create a circular classifier
#2 input nodes for X1 and X2
classifier.add(Dense(4, input_shape = (2, ), activation = 'sigmoid'))
classifier.add(Dense(1, activation = 'sigmoid' ))
classifier.compile(Adam(lr = 0.01), loss = 'binary_crossentropy', metrics= ['accuracy'])
h = classifier.fit(X,y, verbose = 1, batch_size = 20, epochs = 50 , shuffle = True)

plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['accuracy'])