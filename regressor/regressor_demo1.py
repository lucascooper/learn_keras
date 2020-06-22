import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


#  create some data
X = np.linspace(-1,1,200)
np.random.shuffle(X)
Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,))
# plot data
plt.scatter(X,Y)
plt.show()

X_train, Y_train = X[:160],Y[:160]
X_test, Y_test = X[160:],Y[160:]

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))

#  choose loss function and optimizing method
model.compile(loss='mse',optimizer='sgd')

# train
print('Training---------------')
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step%100 == 0:
        print('train cost:',cost)

# test
print('\nTesting-------------')
cost = model.test_on_batch(X_test,Y_test)
print('test cost:',cost)
W,b = model.layers[0].get_weights()
print('Weights=',W,'biases=',b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()