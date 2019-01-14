
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras import Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np

## Data load
direc = 'C:\SONBUKUN\Doosan\Simulation/'
file = 'init_data.csv'
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

    x_train = inputs[:cut_len]
    x_train = np.reshape(x_train,((cut_len), 31,81,1))
    y_train = labels[:cut_len]

    x_test = inputs[cut_len:]
    x_test = np.reshape(x_test, (N-cut_len, 31, 81,1))
    y_test = labels[cut_len:]

    print (y_train[0])

    return x_train, y_train, x_test, y_test


def build_model():
    # build in functional api, not sequential model to merge models

    pcd_in = Input(shape=(31,81,1), name='pcd')
    conv1 = layers.Conv2D(16, 2)(pcd_in)
    batch1 = layers.BatchNormalization()(conv1)
    act1 = layers.Activation(activation='relu')(batch1)
    pool1 = layers.MaxPooling2D((2, 2))(act1)
    conv2 = layers.Conv2D(16, 2, activation='relu')(pool1)
    flat1 = layers.Flatten()(conv2)
    soft1 = layers.Dense(128, activation='softmax')(flat1)
    dense1 = layers.Dense(64,activation='relu')(soft1)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    answer = layers.Dense(3)(dense2)

    final_model = Model(pcd_in, answer)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    final_model.compile(optimizer =adam, loss='mse', metrics=['mae'])
    print (final_model.summary())

    return final_model

if __name__ =='__main__':
    inputs, labels, N, D = dataload()
    x_train, y_train, x_test, y_test = wrangle_data(inputs, labels, N, 0.8)
    model = build_model()

    early_stop = EarlyStopping(patience = 20)
    history = model.fit({'pcd': x_train}, y_train, batch_size=32, epochs=500,
                        validation_split=0.2, shuffle=True, callbacks=[early_stop])

    score = model.evaluate({'pcd': x_test}, y_test)

    print('Loss : ', score[0])
    print('MAE : ', score[1])

    mae_history = history.history['val_mean_absolute_error']
    loss_history = history.history['loss']

    model_json = model.to_json()

    with open("C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results\model_init1.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results\model_init1.h5")

    f1 = plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(range(1,len(mae_history)+1),mae_history)
    plt.title('Model mean abolute error')

    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')

    plt.subplot(1,2,2)
    plt.title('Model training loss')
    plt.plot(range(1, len(mae_history) +1), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')

    f1.show()

    f2 = plt.figure(2)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(mae_history) -9), mae_history[10:])
    plt.title('Model mean abolute error')
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')

    plt.subplot(1, 2, 2)
    plt.title('Model training loss')
    plt.plot(range(1, len(mae_history) -9), loss_history[10:])
    plt.xlabel('Epochs')
    plt.ylabel('MAE Loss')

    f2.show()


