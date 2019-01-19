
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam
from scipy import io
from cyl2q import Cyl2Q
from math import *

model_dir = "C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results/"

if __name__ == '__main__':

    json_file = open(model_dir + "model_doosan4.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir + "model_doosan4.h5")

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loaded_model.compile(optimizer=adam, loss='mae', metrics=['mae'])

    trjExp = io.loadmat("C:\SONBUKUN\Doosan\Data201811\Lraw_fl.mat")['Lraw_fl']


    L1_Exp = trjExp[0][0][0]
    L2_Exp = trjExp[0][0][1]
    L3_Exp = trjExp[0][0][2]

    L1 = L1_Exp[0][0]
    L2 = L2_Exp[0][0]
    L3 = L3_Exp[0][0]

    cyl2q = Cyl2Q()
    q1, q2, q3 = cyl2q.cyl2q(L1, L2, L3)

    mode = 1
    if mode == 1:
        q1 = 5.98999026876939
        q2 = -56.1399844303374
        q3 = 19.9793618757278
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_fl1.csv"

    elif mode == 2:
        q1 = 10.9400024758006
        q2 = -56.1099856510405
        q3 = 20.8774509368948
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_fl2.csv"

    elif mode == 3:
        q1 = -13.4800109519337
        q2 = - 38.8899844303374
        q3 = 5.27794653212942
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_sl1.csv"

    elif mode == 4:
        q1 = -10.4500121726369
        q2 = - 50.4299929752593
        q3 = 2.03707767496352
        pcd_file = "C:\SONBUKUN\Doosan\Data201811\PCD_sl2.csv"

    pcd_read = pd.read_csv(pcd_file)
    pcd_arr = pcd_read.values
    pcd_arr = np.reshape(pcd_arr, (1, 31, 81, 1))
    pcd_arr = pcd_arr + np.random.normal(loc=0.0, scale=0.2, size=(1, 31, 81, 1))

    for i in range(720):
        q = loaded_model.predict({'pcd': pcd_arr, 'joint': np.array([[q1, q2, q3, (i+1)/100.0]])})
        q1 = q[0][0]
        q2 = q[0][1]
        q3 = q[0][2]
        print(q1,'\t',q2,'\t',q3)
        if q3 <= -87:   break






