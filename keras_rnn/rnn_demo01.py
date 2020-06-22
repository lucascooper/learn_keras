import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import SimpleRNN,Dense
from keras.optimizers import Adam
from keras.datasets import mnist


# hyper parameters
LR = 0.001
TIME_STEPS = 28
INPUT_SIZE = 28
EPOCH = 2
BATCH_SIZE = 32
RNN_CELLS = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10

# load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()
# data pre-processing
X_train = X_train.reshape(-1,28,28)/255
X_test = X_test.reshape(-1,28,28)/255
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
print(X_train.shape)

# build RNN
model = Sequential()
model.add(
    SimpleRNN(RNN_CELLS,batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE)
              # ,return_sequences=True,return_state=True
              )
)
model.add(Dense(OUTPUT_SIZE,activation='softmax'))

# define optimizer
adam = Adam(LR)
model.compile(adam,loss='categorical_crossentropy',metrics=['acc'])

# start to train model
print('Training------------------------------')
model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH)
# for i in range(4001):
#     X_batch = X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :, :]
#     y_batch = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :]
#     model.train_on_batch(X_batch,y_batch)
#     BATCH_INDEX += BATCH_SIZE
#     BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

print('Testing-------------------------------')
loss,accuracy = model.test_on_batch(X_test[0:BATCH_SIZE, :, :],y_test[0:BATCH_SIZE, :])
print('test loss:', loss)
print('test accuracy:', accuracy)






