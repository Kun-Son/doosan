
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam
from scipy import io
from cyl2q import Cyl2Q
from math import *

model_dir = "C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results/"

if __name__ == '__main__':

    json_file = open(model_dir + "model_init1.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir + "model_init1.h5")

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loaded_model.compile(optimizer=adam, loss='mae', metrics=['mae'])

    mode = 2
    if mode == 1:
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_fl1.csv"

    elif mode == 2:
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_fl2.csv"

    elif mode == 3:
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_sl1.csv"

    elif mode == 4:
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_sl2.csv"

    pcd_read = pd.read_csv(pcd_file)
    pcd_arr = pcd_read.values
    pcd_arr = np.reshape(pcd_arr, (1, 31, 81, 1))
    pcd_arr = pcd_arr + np.random.normal(loc=0.0, scale=0.2, size=(1, 31, 81, 1))

    q = loaded_model.predict({'pcd': pcd_arr},steps=1)
    print(q[0])






