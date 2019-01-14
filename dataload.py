# -*- coding: utf-8 -*-
"""
author: Bukun Son
created: 2018.01.09
updated: 2018.01.09
"""

import numpy as np
import pandas as pd
import random as rd

class DataLoad():
  def __init__(self, direc, data_file):
    """Init the class
    input:
    - direc: the folder with the datafiles
    - csv_file: the name of the csv file    """
    assert direc[-1] == '/', 'Please provide a directory ending with a /'
    assert data_file[-4:] == '.csv', 'Please provide a filename ending with .csv'
    self.data_loc = direc+data_file    #The location of the csv file
    self.data_raw = []                  #the list where eventually will be all the data

    #After munging, the data_raw will be in [n, seq_len, crd]
    self.labels = []                 #the list where eventually will be all the data
    self.is_abs = True               #Boolean to indicate if we have absolute data or offset data
    self.data = {}                   #Dictionary where the train and val date are located after split_train_test()

  def wrangle_data(self):

    data_read = pd.read_csv(self.csv_loc)
    data_arr = data_read.values

    start_index = 0
    min_step = 456
    N, D = data_arr.shape

    for i in range(N):
        if int(data_arr[i, 3]) == 1:  # Check the end of sequence
            end_index = i  # Note this represent the final index + 1
            # Now we have the start index and end index
            seq = np.array(data_arr[start_index:end_index + 1])  # Cut as a sequence
            self.data_raw.append(rd.sample(list(seq[:i]),min_step))  # Add the current sequence to the list
            self.labels.append(rd.sample(list(seq[1:i + 1,:3]),min_step))
            start_index = end_index
    try:
        self.data_raw = np.stack(self.data_raw,0)
        self.labels = np.stack(self.labels,0)
        self.N = len(self.labels)
    except:
        print('Something went wrong when convert list to np array')
    print('Data wrangling is done.')

  def split_train_test(self, ratio):
    assert not isinstance(self.data_raw, list), 'First wrangle the data before returning'
    print (self.data_raw.shape)
    N,seq_len,crd = self.data_raw.shape
    #assert seq_len > 1, 'Seq_len appears to be singleton'
    assert ratio <= 1.0, 'Provide ratio as a float between 0 and 1'
    #Split and shuffle the data
    ind_cut = int(ratio*N)
    #ind = np.random.permutation(N)

    self.data['X_train'] = self.data_raw[:ind_cut]
    self.data['y_train'] = self.labels[:ind_cut]
    self.data['X_val'] = self.data_raw[ind_cut:]
    self.data['y_val'] = self.labels[ind_cut:]
    print('%.0f Train samples and %.0f val samples' %(ind_cut, N-ind_cut))
    return

  def return_data_list(self,ratio,ret_list = True):
    ## From data_raw in [N,seq_len,crd] returns a list of [N,crd] with seq_len elements
    assert not isinstance(self.data_raw, list), 'First wrangle the data before returning'
    N,seq_len,crd = self.data_raw.shape
    assert seq_len > 1, 'Seq_len appears to be singleton'
    assert ratio <= 1.0, 'Provide ratio as a float between 0 and 1'
    data = {}  #dictionary for the data
    #Split and Shuffle the data
    ind_cut = int(ratio*N)
    #ind = np.random.permutation(N)

    data['X_train'] = self.data_raw[:ind_cut]
    data['X_val'] = self.data_raw[ind_cut:]
    data['y_train'] = self.labels[:ind_cut]
    data['y_val'] = self.labels[ind_cut:]

    if ret_list:  #Do you want train data as list or 3D np array
      for key in ['X_train','X_val']:
        listdata = []
        for i in range(seq_len):
          listdata.append(data[key][:,i,:])
        data[key] = listdata
    print('Returned data')

    return data

