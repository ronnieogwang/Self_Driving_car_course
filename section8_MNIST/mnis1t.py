'''
Data is split into;
1. Training set- data with label, the NN learns on this data, tune standard params
2. Validation set- used to tune hyperparameters
3. Test set- not labelled, used to test 

**Hyper parameraters**
-learning rate
-nodes per layer
-number of hidden layers
 '''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical 
import random

#seed random
np.random.seed(0)
'''Part 1, get and display dataset'''
#get dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#verify dataset, if condition is met, code executes else stops
assert(X_train.shape[0] == y_train.shape[0]), "The number of images not equal to labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images not equal to labels"
assert(X_train.shape[1:] == (28,28)), "Dimensions not 28 x 28"
assert(X_test.shape[1:] == (28,28)), "Dimensions not 28 x 28"

#visualize images to know how many classes
num_of_samples = []

cols = 5 #grid of 5 
classes = 10

#grid
fig, axs = plt.subplots(nrows = classes, ncols = cols, figsize = (5,10))  #subplots allows to plot many objects on one page, returns tuple 
fig.tight_layout()     #removes overlap
for i in range(cols):
    for j in range(classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected-1)), :, :], cmap = plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
        
#show bar chart for images per class
plt.figure(figsize=(12,4))
plt.bar(range(0,classes), num_of_samples)
plt.title("Distributin of data")
plt.xlabel('classes')
plt.ylabel('num of images') 

'''Part 2 Data preprocessing'''

#onehot encoding
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

#normalization to in a range 0 to 1, reduces variance in the data
X_train = X_train/255
X_test = X_test/255

#flatten images
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

#creating an ANN to classify data
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim = num_pixels, activation = 'relu'))
    model.add(Dense(10,  activation = 'relu'))
    model.add(Dense(classes, input_dim = num_pixels, activation = 'softmax'))
    model.compile(Adam(lr =0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
model = create_model()
model.summary()

#train
h =model.fit(X_train, y_train, validation_split = 0.1, epochs = 10, batch_size = 200, verbose =1, shuffle= 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend('loss', 'valloss')

#get test image from net
import requests
from PIL import Image
 
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

#convert image to array, then to 28 x 28 gray scale
import cv2
img = np.asarray(img)
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)# change white to black and viceverser
plt.imshow(img, cmap=plt.get_cmap('gray'))#remove color

img = img/255
img = img.reshape(1, 784)
prediction = model.predict_classes(img)

