from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking,Embedding,Reshape,Flatten
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
TestLength=1000
batch=50

filedir='/data/biantian/data/new_tfidf_dataset'
def generate_arrays_from_file(path,batch_size):
    global filelist
    while 1:
        #print(totalcnt)
        x_data = []
        index = 0
        y_data_train = [0] * batch_size
        X = []
        Y = []
        #for filename in os.listdir(path)[0:3000]:
        # if TestLength+totalcnt==TrainLength+TestLength:
        #     totalcnt=0
        for filename in filelist[TestLength:TestLength+TrainLength]:
            f = open(str(path + '/' + filename), encoding='utf-8')
            y_data_train[index] = float(f.readline())
            x = f.readlines()
            new_x = []
            for cur_x in x:
                cur_x = cur_x.replace("[", "").replace("]", "").split(",")
                temp_x = []
                for floatx in cur_x:
                    temp_x.append(float(floatx))
                new_x.append(temp_x)
            if len(new_x) == 0:
                continue
            x_data.append(new_x)
            index = index + 1
            #print("x:",index)

            if index == batch_size:
                # 通过补充-1将长度统一
                # X_data=[]
                tempx = 0
                global maxcount
                count = [0] * batch_size
                #print("training maxcount:", maxcount)
                for xdata in x_data:
                    tempx = tempx + 1
                    #print("train_x:", tempx)
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
                x_data=[]
                X = X_data
                Y= np_utils.to_categorical(y_data_train, num_classes=2)
                index = 0
                yield (X, Y)
                X = []
                Y = []
                #break
            f.close()


#给数据打乱
index = [i for i in range(len(os.listdir(filedir)))]
np.random.shuffle(index)
indices=0
filelist=[0]*len(os.listdir(filedir))
for tempindex in index:
    filelist[indices] = os.listdir(filedir)[tempindex]
    indices=indices+1

y_data_test = [0]*TestLength
Test_x_data = []
Test_index = 0
Test_count = [0] * TestLength
for filename in filelist[0:TestLength]:
    f = open(str(filedir + '/' + filename), encoding='utf-8')
    y_data_test[Test_index]=float(f.readline())
    x=f.readlines()
    #x = f.read().replace("[", "").replace("]", "").replace("\n", "\\\\")
    #x = x.split("\\\\")
    #x.pop()
    new_x = []
    for cur_x in x:
        Test_count[Test_index] = Test_count[Test_index] + 1
        cur_x = cur_x.replace("[", "").replace("]", "").split(",")
        temp_x = []
        for floatx in cur_x:
            temp_x.append(float(floatx))
        new_x.append(temp_x)
    f.close()
    if len(new_x) == 0:
        print(filename)
        continue
    Test_x_data.append(new_x)
    Test_index = Test_index + 1
    #print(Test_index)
    if Test_index == TestLength:
        break
# 通过补充-1将长度统一
# X_data=[]
tempx = 0
#print("Test_x:",len(Test_x_data))
#print(Test_x_data[1][1])
for xdata in Test_x_data:
    tempx = tempx + 1
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
#print("test maxcount:",maxcount)

#定义测试集标签
y_data = np_utils.to_categorical(y_data_test, num_classes=2)
y_test = y_data[0:TestLength, :]



model = Sequential()
model.add(Masking(mask_value= -1,input_shape=(maxcount, 5000,)))
# print(model.output_shape)
# model.add(Embedding(5000,100,input_length=95))
# print(model.output_shape)
model.add(Dense(100))
model.add(GRU(100,return_sequences=True))
model.add(GRU(100))
model.add(Dense(2, activation='softmax'))

adagrad = Adagrad(LR)
model.compile(optimizer=adagrad,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file(filedir,batch_size=batch),validation_data=(x_test, y_test),epochs=10,steps_per_epoch=60)
#model.fit(x_train, y_train, batch_size=100, epochs=10)
#score,acc = model.evaluate(x_test, y_test, batch_size=batch)
#print('score:', score)
#print('acc:', acc)
