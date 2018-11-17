# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import SimpleRNN,Dense,Masking
from keras.optimizers import Adagrad

np.random.seed(7)  # for reproducibility
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
    if len(new_x) == 0:
        continue
    x_data.append(new_x)
    index=index+1
    if index==200:
        break
#通过补充-1将长度统一
X_data=[]
for xdata in x_data:
    if len(xdata)<max(count):
        for i in range(max(count)-len(xdata)):
            xdata.append([-1]*5000)
    X_data.append(xdata)
X_data=np.array(X_data)
x_train,x_test=X_data[0:150,:],X_data[150:200,:]

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
y_train,y_test=y_data[0:150,:],y_data[150:200,:]

model = Sequential()
model.add(Masking(mask_value= -1,input_shape=(max(count), 5000,)))
# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    #batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    batch_input_shape=(None, max(count), 5000),
    output_dim=100,
    #unroll=True,
))

# output layer
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=10)
score,acc = model.evaluate(x_test, y_test, batch_size=50)
print('score:', score)
print('acc:', acc)
