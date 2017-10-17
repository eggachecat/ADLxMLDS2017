"""
    1. change label to one-hat
    2. input from mfcc and truth from label from 1.
    3. train the network which can predict syllables give frame
    4. convert syllables to phone
    5. use edit_distance to define the cost function: edit_distance(phone, truth)
    6. train
    
    *** some issue exists in 3. ~ 6.
        1. number of frames are different in each piece of voice!
        2. 
        
    Maybe we can first train mlp to 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def translate_data(src, dst, delimiter=" "):
    df = pd.read_csv(src, header=None, delimiter=delimiter)
    df.to_hdf(dst, 'train', mode='w')


# translate_data("./data/fbank/train.ark", "./data/fbank/train.hdf5")
# translate_data("./data/label/train.lab", "./data/label/train.hdf5", ",")


def read_data(src):
    df = pd.read_hdf(src, 'train')
    return df


def run_fbank():
    translate_data("./data/test/train.ark", "./data/test/train.hdf5")
    translate_data("./data/fbank/train.ark", "./data/fbank/train.hdf5")
    translate_data("./data/label/train.lab", "./data/label/train.hdf5", ",")

    df_1 = read_data("./data/fbank/train.hdf5")
    df_2 = read_data("./data/label/train.hdf5")

    df = pd.merge(df_1, df_2, left_on=0, right_on=0)
    df.to_hdf("./data/train.hdf5", 'train', mode='w')


def run_mfcc():
    translate_data("./data/mfcc/test.ark", "./data/test_mfcc.hdf5")
    exit()
    translate_data("./data/mfcc/train.ark", "./data/mfcc/train.hdf5")
    translate_data("./data/label/train.lab", "./data/label/train.hdf5", ",")

    df_1 = read_data("./data/mfcc/train.hdf5")
    df_2 = read_data("./data/label/train.hdf5")

    df = pd.merge(df_1, df_2, left_on=0, right_on=0)
    df.to_hdf("./data/train_mfcc.hdf5", 'train', mode='w')


def to_one_hot(_labels):
    _unique_labels = np.unique(_labels)
    _enc = OneHotEncoder()
    _enc.fit(_unique_labels.reshape(-1, 1))
    return _enc.transform(_labels.reshape(-1, 1)).toarray()


def translate_merged_data():
    _df = pd.read_hdf("./data/train.hdf5", 'train')

    _48phone_map = pd.read_csv("./data/48phone_char.map", delimiter="\t", header=None)
    _48phone_map.set_index(0, drop=True, inplace=True)
    _48phone_map = _48phone_map[[1]].to_dict()[1]
    _df[_df.columns[-1]] = _df[_df.columns[-1]].apply(lambda x: _48phone_map[x])

    # split to time step!
    _uid, _sid, _fid = _df[0].str.split('_').str
    _df["iid"], _df["ts"] = _uid + "_" + _sid, _fid.apply(int)
    iid_list = _df["iid"].unique()

    _df.set_index(0, drop=True, inplace=True)
    data_table = _df.as_matrix()
    data_table = data_table[data_table[:, -1].argsort()]
    data_table_x = np.asarray([np.asarray(data_table[data_table[:, -2] == iid, :-3]) for iid in iid_list])
    data_table_y = np.asarray([np.asarray(data_table[data_table[:, -2] == iid, -3]) for iid in iid_list])
    print("begin saving data..")
    np.save("./data/train_data_x.npy", data_table_x)
    np.save("./data/train_data_y.npy", data_table_y)

    exit()

    x_data = _df[_df.columns[0:-1]].as_matrix()
    y_data = to_one_hot(_df[_df.columns[-1]].as_matrix())

    return x_data, y_data


# translate_merged_data()
run_mfcc()
# data_table = np.load("./data/train_data.npy")
# print(data_table)
