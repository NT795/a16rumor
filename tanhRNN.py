# -*- coding: utf-8 -*-
import numpy as np
import os
np.random.seed(7)  # for reproducibility

from keras.utils import np_utils
from keras.layers import SimpleRNN, Activation, Dense,Input
from keras.optimizers import Adagrad
from keras.models import Model

    # same as the height of the image
INPUT_SIZE = 5000     # same as the width of the image
BATCH_SIZE = 1
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 100
#TIME_STEPS = 50
LR = 0.5

filedir='F:\论文汇总\谣言检测\数据\A16rumdect\dataset'

x_data=[]
index=0
count=[0]*200
for filename in os.listdir(filedir):
    f = open(str(filedir + '/' + filename),encoding='utf-8')
    x = f.read().replace("[","").replace("]","").replace("\n","\\\\")
    x=x.split("\\\\")
    x.pop()
    new_x=[]
    for cur_x in x:
        count[index]=count[index]+1
        cur_x=cur_x.split(",")
        temp_x=[]
        for floatx in cur_x:
            temp_x.append(float(floatx))
        new_x.append(temp_x)
    f.close()
    x_data.append(new_x)
    index=index+1
    if index==20:
        break


f1 = open('F:\论文汇总\谣言检测\数据\A16rumdect\Weibo.txt', encoding='utf-8')
y = f1.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data=[]
for cur_y in y:
    cur_y=cur_y.split(" ")
    new_y=cur_y[0].split("\t")
    if len(new_y)==3:
        new1_y=new_y[1].split(":")
        y_data.append(new1_y[1])
f1.close()
y_data = np_utils.to_categorical(y_data[0:200], num_classes=2)

#X_train,X_test=x_data[0:1500,:],x_data[1500:2000,:]
y_train,y_test=y_data[0:15,:],y_data[15:20,:]

"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train)
print(y_train)

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
print(X_train)
print(y_test)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build RNN model
model = Sequential()
# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    #unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
"""
main_input = Input(shape=(5000,))
step_input = Input(shape=(1,))
# a layer instance is callable on a tensor, and returns a tensor
x = SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, step_input, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    #unroll=True,
)(main_input)
result = Dense(OUTPUT_SIZE, activation='softmax')(x)


# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=[main_input,step_input], outputs=result)

# optimizer
adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

# training
for step in range(16):
    # data shape = (batch_num, steps, inputs/outputs)
    #X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    X_batch=x_data[step]
    X_batch=np.array(X_batch)
    #X_batch=X_batch.reshape(1,X_batch.shape[0],X_batch.shape[1])
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    #TIME_STEPS=count[step]
    cost = model.train_on_batch([X_batch,count], Y_batch)
    BATCH_INDEX += BATCH_SIZE
    #BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    BATCH_INDEX = 0 if BATCH_INDEX >= 15 else BATCH_INDEX
    if step % 3 == 0:
        TIME_STEPS=count[step/3+14]
        #cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        cost, accuracy = model.evaluate(x_data[step/3+14], y_test, batch_size=1, verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)