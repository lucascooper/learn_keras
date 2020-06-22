import numpy as np
np.random.seed(1332)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


# load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# normalize and one hot encode
X_train = X_train.reshape(X_train.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# build model
model = Sequential([
    Dense(32, activation='relu', input_dim=28*28),
    Dense(10, activation='softmax')
])

# build RMSProp optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

# compile the model
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['acc'])

# fit model
print('Training-------------------------------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

#
print('\nTesting---------------------------')
loss,accuracy = model.test_on_batch(X_test,y_test)

print('loss:', loss)
print('accuracy:',accuracy)
