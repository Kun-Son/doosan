
import rospy
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam
from scipy import io
from cyl2q import Cyl2Q
from math import *
from PythonSocketCAN import PythonSocketCAN

Bm_init = 60.34
Am_init = -121.43
Bk_init = -0.5

Bm_fin =  30
Am_fin = -80
Bk_fin = -100

model_dir = "C:\SONBUKUN\Doosan\Simulation\CNN_MLP-VX\\results/"

def callback():
    rospy.loginfo(rospy.get_caller_id() + "Received PCD data.")

def listner():
    rate = rospy.Rate(10)
    input("Press Enter to first position.")
    flag = 1

    while True:
        rospy.Subscriber("config", String, callback)
        bm, am, bk = asdlfkjasdklfj

        if flag == 1:
            bm0 = bm
            am0 = am
            bk0 = bk
            incre = [Bm_init-bm0, Am_init-am0, Bk_init-bk0]/200.0

        flag = first_pos(bm, am, bk, incre)
        if flag == 3:    break
        rate.sleep()

    input("Press Enter to dig.")
    while True:
        rospy.Subscriber("config", String, callback)
        bm, am, bk = asdlfkjasdklfj

    while True:
        rospy.Subscriber("config", String, callback)
        bm, am, bk = asdlfkjasdklfj

    # digging process
    c = 0.0
    while True:
        rospy.Subscriber("config", String, callback)
        if c >= 7.2 or bm <= -87:   flag = 3
        else:

            c = c + 0.01


def load_initmodel():
    json_file = open(model_dir + "model_init1.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir + "model_init1.h5")

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loaded_model.compile(optimizer=adam, loss='mae', metrics=['mae'])

    return loaded_model

def load_exmodel():
    json_file = open(model_dir + "model_doosan4.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir + "model_doosan4.h5")

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loaded_model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    return loaded_model

def first_pos(bm, am, bk, incre):
    tar = np.array[Bm_init, Am_init, Bk_init]
    now = np.array[bm, am, bk]

    if np.linalg.norm(tar-now)<=3:  flag = 3
    else:
        PythonSocketCAN.can_producer(0x0CFF0461,bm+incre[0],am+incre[1],bk+incre[2])
        flag = 2

    return flag

if __name__ == '__main__':

    rospy.init_node('Controller', anonymous=True)
    init_model = load_initmodel()
    xcv_model = load_exmodel()

    PythonSocketCAN(channel='can3',bustype='socketcan_native',bitrate = '500000')
    listner()



    #raw_input("Press Enter to initializing")   #in case of python2.7


