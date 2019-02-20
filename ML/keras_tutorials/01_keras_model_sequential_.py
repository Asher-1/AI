from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()



