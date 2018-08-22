from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding,Masking
from keras.layers import GRU
from keras.utils import np_utils
from keras.optimizers import Adagrad
import os
import numpy as np

     # same as the height of the image
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
#model.add(Embedding(5000,100))
model.add(GRU(100))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=10)
score,acc = model.evaluate(x_test, y_test, batch_size=50)
print('score:', score)
print('acc:', acc)
"""
total_cost = 0
total_accuray = 0
for step in range(150):
    # data shape = (batch_num, steps, inputs/outputs)
    #X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    X_batch=x_data[step]
    X_batch=np.array(X_batch)
    #count=np.array(count)
    #X_batch=X_batch.reshape(1,X_batch.shape[0],X_batch.shape[1])
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    #TIME_STEPS=count[step]
    #cost = model.train_on_batch([X_batch,TIME_STEPS], Y_batch)
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    #BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    BATCH_INDEX = 0 if BATCH_INDEX >= 150 else BATCH_INDEX
    if (step+1)% 15==0:
        #print("xstep",step)
        print('test cost: ', total_cost/5, 'test accuracy: ', total_accuray/5)
        total_cost=0
        total_accuray=0
    if step % 3 == 0 and step>0:#相当于5个一个batch
        y_STEPS=int(step/3+149)
        X_test = np.array(x_data[y_STEPS])
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        Y_test=y_test[y_STEPS-150]
        #print("ystep",y_STEPS-150)
        Y_test=Y_test.reshape(1, Y_test.shape[0])
        #cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=1, verbose=False)
        total_cost=total_cost+cost
        total_accuray=total_accuray+accuracy
        #print('test cost: ', cost, 'test accuracy: ', accuracy)
"""