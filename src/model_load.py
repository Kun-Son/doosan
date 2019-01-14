
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np

## Data load
direc = 'C:\SONBUKUN\Doosan\Simulation/'
file = 'dataset1.csv'

model_dir = "C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results/"
xnum = 31
ynum = 81
dof = 3

def dataload():
    assert direc[-1] == '/', 'Directory ending with a /'
    assert file[-4:] == '.csv', 'Please provide a filename ending with .csv'

    csv_loc = direc + file
    data_read = pd.read_csv(csv_loc)
    data_arr = data_read.values

    N, D = data_arr.shape

    np.random.shuffle(data_arr)

    inputs = data_arr[:,:-dof]
    labels = data_arr[:,-dof:]

    print('Dataset size is (%d, %d)' %(N,D))

    return inputs, labels, N, D


def wrangle_data(inputs, labels, N, ratio):

    cut_len = int(N * ratio)

    x1_train = inputs[:cut_len, : -(dof+1)]
    x1_train = np.reshape(x1_train,((cut_len), 81,31,1))
    x2_train = inputs[:cut_len, -(dof+1) : ]
    y_train = labels[:cut_len]

    x1_test = inputs[cut_len:, : -(dof + 1)]
    x1_test = np.reshape(x1_test, (N-cut_len, 81, 31,1))
    x2_test = inputs[cut_len:, -(dof + 1):]
    y_test = labels[cut_len:]

    return x1_train, x2_train, y_train,  x1_test, x2_test, y_test


if __name__ =='__main__':
    inputs, labels, N, D = dataload()
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = wrangle_data(inputs, labels, N, 0.8)


    json_file = open(model_dir+"model_doosan.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir+"model_doosan.h5")

    loaded_model.compile(optimizer ='rmsprop', loss='mse', metrics=['mae'])
    score = loaded_model.evaluate({'pcd': x1_test, 'joint': x2_test}, y_test)

    print('Loss : ', score[0])
    print('MAE : ', score[1])

