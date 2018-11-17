from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import GRU
from keras.utils import np_utils
from keras.optimizers import Adagrad
import os
import numpy as np
     # same as the height of the image
np.random.seed(7)  # for reproducibility
LR = 0.5
maxcount=95
TrainLength=3000
TestLength=1500
batch=100

filedir='/data/biantian/data/dataset'
def generate_arrays_from_file(path,batch_size):
    while 1:
        X = []
        Y = []
        x_data = []
        index = 0
        count = [0] * batch_size
        epochy = 0
        #for filename in os.listdir(path)[0:3000]:
        for filename in os.listdir(path)[0:TrainLength]:
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
                maxcount = max(max(count), maxcount)
                print("training maxcount:", maxcount)
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
                x_data=[]
                count = [0] * batch_size
                fy = open('/data/biantian/data/Weibo.txt', encoding='utf-8')
                y = fy.read().replace("\n", "\\\\")
                y = y.split("\\\\")
                y_data = []
                tempy = 0
                cnt = 0
                for cur_y in y:
                    tempy=tempy+1
                    #if tempy>3000:
                    if tempy > TrainLength:
                        break
                    if tempy<=epochy:
                        continue
                    print("train_y:", tempy)
                    cur_y = cur_y.split(" ")
                    new_y = cur_y[0].split("\t")
                    if len(new_y) == 3:
                        cnt = cnt + 1
                        new1_y = new_y[1].split(":")
                        y_data.append(new1_y[1])
                    if cnt == batch_size:
                        epochy = epochy + cnt
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
Test_count = [0] * TestLength
for filename in os.listdir(filedir)[TrainLength:TrainLength+TestLength+1]:
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
    if Test_index == TestLength:
        break
# 通过补充-1将长度统一
# X_data=[]
tempx = 0
print("Test_x:",len(Test_x_data))
for xdata in Test_x_data:
    tempx = tempx + 1
    print("test_x:", tempx)
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
x_test = X_data[0:TestLength, :]
maxcount=max(max(Test_count),maxcount)
print("test maxcount:",maxcount)

f1 = open('/data/biantian/data/Weibo.txt', encoding='utf-8')
y = f1.read().replace("\n", "\\\\")
y = y.split("\\\\")
y_data = []
tempy = 0
for cur_y in y:
    tempy = tempy + 1
    if tempy<=TrainLength:
        continue
    if tempy>TrainLength+TestLength:
        break
    print("test_y:", tempy)
    cur_y = cur_y.split(" ")
    new_y = cur_y[0].split("\t")
    if len(new_y) == 3:
        new1_y = new_y[1].split(":")
        y_data.append(new1_y[1])
f1.close()
y_data = np_utils.to_categorical(y_data[0:TestLength], num_classes=2)
y_test = y_data[0:TestLength, :]


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

model.fit_generator(generate_arrays_from_file(filedir,batch_size=batch),validation_data=(x_test, y_test),epochs=10,steps_per_epoch=30)
#model.fit(x_train, y_train, batch_size=100, epochs=10)
#score,acc = model.evaluate(x_test, y_test, batch_size=batch)
#print('score:', score)
#print('acc:', acc)
