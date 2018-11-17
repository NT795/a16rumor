from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import GRU
from keras.utils import np_utils
from keras.optimizers import Adagrad
import os
import numpy as np
"""
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
#model.add(Flatten())
model.add(Dense(100))
model.add(GRU(100,return_sequences=True))
model.add(GRU(100))
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=10)
score,acc = model.evaluate(x_test, y_test, batch_size=50)
print('score:', score)
print('score:', acc)
"""
     # same as the height of the image
np.random.seed(7)  # for reproducibility
LR = 0.5
maxcount=95
filedir='F:\论文汇总\谣言检测\数据\A16rumdect\dataset'
def generate_arrays_from_file(path,batch_size):
    while 1:
        X = []
        Y = []
        x_data = []
        index = 0
        count = [0] * batch_size
        epochy = 0
        for filename in os.listdir(path)[0:300]:
            fx = open(str(path + '/' + filename), encoding='utf-8')
            x = fx.read().replace("[", "").replace("]", "").replace("\n", "\\\\")
            x = x.split("\\\\")
            x.pop()
            new_x = []
            for cur_x in x:
                count[index] = count[index] + 1
                cur_x = cur_x.split(",")
                temp_x = []
                for floatx in cur_x:
                    temp_x.append(float(floatx))
                new_x.append(temp_x)
            if len(new_x) == 0:
                continue
            x_data.append(new_x)
            index = index + 1
            if index == batch_size:
                #break
                index=0
                # 通过补充-1将长度统一
                # X_data=[]
                tempx = 0
                global maxcount
                maxcount=max(max(count),maxcount)
                #print("training maxcount:",maxcount)
                for xdata in x_data:
                    tempx = tempx + 1
                    print("train_x:", tempx)
                    if len(xdata) < maxcount:
                        for i in range(maxcount - len(xdata)):
                            xdata.append([-1] * 5000)
                    X_data_temp = np.array(xdata)
                    X_data_temp = X_data_temp.reshape(1, X_data_temp.shape[0], X_data_temp.shape[1])
                    if tempx == 1:
                        X_data = X_data_temp
                        continue
                    X_data = np.vstack((X_data, X_data_temp))
                    # X_data.append(xdata)
                # X_data=np.array(X_data)
                # x_train, x_test = X_data[0:1000, :], X_data[1000:1500, :]
                X = X_data
                X_data=[]
                x_data=[]
                count = [0] * batch_size
                fy = open('F:\论文汇总\谣言检测\数据\A16rumdect\Weibo.txt', encoding='utf-8')
                y = fy.read().replace("\n", "\\\\")
                y = y.split("\\\\")
                y_data = []
                tempy = 0
                cnt = 0
                for cur_y in y:
                    tempy = tempy + 1
                    if tempy>300:
                        break
                    if tempy<=epochy:
                        continue
                    cur_y = cur_y.split(" ")
                    new_y = cur_y[0].split("\t")
                    if len(new_y) == 3:
                        cnt = cnt + 1
                        print("train_y:", tempy)
                        new1_y = new_y[1].split(":")
                        y_data.append(new1_y[1])
                    if cnt == batch_size:
                        epochy = epochy + cnt
                        print("epochy:",epochy)
                        cnt = 0
                        Y = np_utils.to_categorical(y_data[0:batch_size], num_classes=2)
                        yield (X, Y)
                        break
            X = []
            Y = []
    fx.close()
    fy.close()

Test_x_data = []
Test_index = 0
Test_count = [0] * 150
for filename in os.listdir(filedir)[300:450]:
    f = open(str(filedir + '/' + filename), encoding='utf-8')
    x = f.read().replace("[", "").replace("]", "").replace("\n", "\\\\")
    x = x.split("\\\\")
    x.pop()
    new_x = []
    for cur_x in x:
        Test_count[Test_index] = Test_count[Test_index] + 1
        cur_x = cur_x.split(",")
        temp_x = []
        for floatx in cur_x:
            temp_x.append(float(floatx))
        new_x.append(temp_x)
    f.close()
    if len(new_x) == 0:
        continue
    Test_x_data.append(new_x)
    Test_index = Test_index + 1
    #print(Test_index)
    if Test_index == 150:
        break
# 通过补充-1将长度统一
# X_data=[]
tempx = 0
for xdata in Test_x_data:
    tempx = tempx + 1
    print("test_x:", tempx)
    #if len(xdata) <max(Test_count):
        #for i in range(max(Test_count) - len(xdata)):
    if len(xdata) < maxcount:
        for i in range(maxcount - len(xdata)):
            xdata.append([-1] * 5000)
    X_data_temp = np.array(xdata)
    X_data_temp = X_data_temp.reshape(1, X_data_temp.shape[0], X_data_temp.shape[1])
    if tempx == 1:
        X_data = X_data_temp
        continue
    X_data = np.vstack((X_data, X_data_temp))
    # X_data.append(xdata)
# X_data=np.array(X_data)
x_test = X_data[0:150, :]
maxcount=max(maxcount,max(Test_count))
#print("test maxcount:",maxcount)
f1 = open('F:\论文汇总\谣言检测\数据\A16rumdect\Weibo.txt', encoding='utf-8')
y = f1.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data = []
tempy = 0
for cur_y in y:
    tempy = tempy + 1
    if tempy<=300:
        continue
    if tempy>450:
        break
    print("test_y:", tempy)
    cur_y = cur_y.split(" ")
    new_y = cur_y[0].split("\t")
    if len(new_y) == 3:
        new1_y = new_y[1].split(":")
        y_data.append(new1_y[1])
f1.close()
y_data = np_utils.to_categorical(y_data[0:150], num_classes=2)
y_test = y_data[0:150, :]

model = Sequential()
model.add(Masking(mask_value= -1,input_shape=(maxcount, 5000,)))
#model.add(Embedding(5000,100))
#model.add(Flatten())
model.add(Dense(100))
model.add(GRU(100,return_sequences=True))
model.add(GRU(100))
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file(filedir,batch_size=10),validation_data=(x_test, y_test),epochs=10,steps_per_epoch=30)
#model.fit(x_train, y_train, batch_size=100, epochs=10)
score,acc = model.evaluate(x_test, y_test, batch_size=10)
print('score:', score)
print('score:', acc)