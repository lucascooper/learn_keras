import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten
from keras.utils import np_utils
from keras.models import Sequential


# load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# start to build graph
model = Sequential()

# Conv layer1 output shape (32,28,28) --> (32,14,14)
model.add(Conv2D(32,5,padding='same',input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Activation('relu'))
# Conv layer2
model.add(Conv2D(64,3,padding='same'))
model.add(MaxPooling2D(strides=(2,2),padding='same'))
model.add(Activation('relu'))

# Flatten and FC
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(10,activation='softmax'))

# define optimizer
adam = Adam(epsilon=1e-8)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])

# start to train
print('Training---------------------------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

# test
loss,accuracy = model.test_on_batch(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)
