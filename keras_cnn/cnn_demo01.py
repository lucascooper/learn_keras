import numpy as np
np.random.seed(1337)
from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


# load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# start to build model
model = Sequential()

# Conv layer 1 output shape (32,28,28)
model.add(Conv2D(
    filters=32,
    kernel_size=5,
    strides=(1,1),
    padding='same',
    input_shape=(1,28,28)  # input_shape(channel,height,width)

))
model.add(Activation('relu'))

# Pooling layer1
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

# Conv layer2 output shape (64,14,14)
model.add(Conv2D(64,5,padding='same'))
model.add(Activation('relu'))

# Pooling layer2 (max pool) output shape (64,7,7)
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# FC
model.add(Flatten())
model.add(Dense(1024,activation='relu'))

model.add(Dense(10,activation='softmax'))

adam = Adam(epsilon=1e-8)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])

# start to fit model
print('Training------------------------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

# test model
print('\nTesting---------------------------')
loss, accuracy = model.test_on_batch(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)
