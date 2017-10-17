import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import keras
import time
import itertools

from keras.preprocessing import sequence


def get_phone_idx_map(data_dir="./data"):
    _48phone_idx_map = pd.read_csv(os.path.join(data_dir, "48phone_char.map"), header=None, delimiter="\t")
    _48phone_idx_map.set_index(0, drop=True, inplace=True)
    phone_idx_map = _48phone_idx_map[[1]].to_dict()[1]
    phone_char_map = _48phone_idx_map[[2]].to_dict()[2]

    return phone_idx_map, phone_char_map


def get_idx_phone_map(data_dir="./data"):
    phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)
    idx_phone_map = dict([(phone_idx_map[k], k) for k in phone_idx_map])
    return idx_phone_map, phone_char_map


def get_raw_label(data_dir="./data"):
    label_dir = os.path.join(data_dir, "label")
    label_path = os.path.join(label_dir, "train.lab")
    return pd.read_csv(label_path, header=None, delimiter=",")


def to_timestep_seq(df):
    _uid, _sid, _fid = df[0].str.split('_').str
    df["iid"], df["ts"] = _uid + "_" + _sid, _fid.apply(int)
    split_indices = np.cumsum(df.groupby("iid").ts.count())

    seq = np.split(df[df.columns[1:-2]].as_matrix(), split_indices)

    return seq


def edit_distance(S1, S2):
    a = len(S1)
    b = len(S2)
    fdn = {}  # Global dict
    for x in range(a + 1):
        fdn[x, 0] = x
    for y in range(b + 1):
        fdn[0, y] = y

    for x in range(1, a + 1):
        for y in range(1, b + 1):
            if S1[x - 1] == S2[y - 1]:
                c = 0
            else:
                c = 1
            fdn[x, y] = min(fdn[x, y - 1] + 1, fdn[x - 1, y] + 1, fdn[x - 1, y - 1] + c)
    return fdn[x, y]


def trim(phone_list, sli="L"):
    s = 0
    while phone_list[s] == sli:
        if s < len(phone_list) - 1:
            s += 1

    e = len(phone_list) - 1
    while phone_list[e] == sli:
        if e > 1:
            e -= 1

    phone_list = phone_list[s:e + 1]
    return [k for k, g in itertools.groupby(phone_list)]


def _get_data(src_type, data_dir="./data", seq=True):
    src_dir = os.path.join(data_dir, src_type)

    train_path = os.path.join(src_dir, "train.ark")
    test_path = os.path.join(src_dir, "test.ark")

    fast_train_path = os.path.join(src_dir, "train.npy")
    fast_test_path = os.path.join(src_dir, "test.npy")

    fast_train_seq_path = os.path.join(src_dir, "train_seq.npy")
    fast_test_seq_path = os.path.join(src_dir, "test_seq.npy")

    try:
        if seq:
            train_seq_stack = np.load(fast_train_seq_path)[()]
            test_seq_stack = np.load(fast_test_seq_path)
        else:
            train_data_stack = np.load(fast_train_path)[()]
            test_data_stack = np.load(fast_test_path)
    except:

        phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)

        raw_train_data = pd.read_csv(train_path, header=None, delimiter=" ")
        train_label = get_raw_label(data_dir)
        train_label[1] = train_label[1].apply(lambda x: phone_idx_map[x])

        train_data = pd.merge(raw_train_data, train_label, left_on=0, right_on=0)

        train_data_stack = {"x": np.array(train_data[train_data.columns[1:-1]].as_matrix()),
                            "y": np.array(train_data[train_data.columns[-1]].as_matrix())}
        raw_test_data = pd.read_csv(test_path, header=None, delimiter=" ")
        test_data_stack = raw_test_data[raw_test_data.columns[1:]].as_matrix()

        if seq:
            raw_train_seq = to_timestep_seq(train_data)
            test_seq_stack = to_timestep_seq(raw_test_data)
            train_seq_stack = {"x": np.array([seq[:, :-1] for seq in raw_train_seq]),
                               "y": np.array([seq[:, -1] for seq in raw_train_seq])}

        try:
            np.save(fast_train_path, train_data_stack)
            np.save(fast_test_path, test_data_stack)
            if seq:
                np.save(fast_train_seq_path, train_seq_stack)
                np.save(fast_test_seq_path, test_seq_stack)
        except:
            print("Could not save!")

    if seq:
        return train_seq_stack, test_seq_stack
    else:
        return train_data_stack, test_data_stack


def get_data_mfcc(data_dir="./data", seq=True):
    return _get_data("mfcc", data_dir, seq)


def get_data_fbank(data_dir="./data", seq=True):
    return _get_data("fbank", data_dir, seq)


def get_data_full(data_dir="./data", seq=True):
    fast_train_path = os.path.join(data_dir, "full_train.npy")
    fast_test_path = os.path.join(data_dir, "full_test.npy")

    fast_train_seq_path = os.path.join(data_dir, "full_train_seq.npy")
    fast_test_seq_path = os.path.join(data_dir, "full_test_seq.npy")

    try:
        if seq:
            train_seq_stack = np.load(fast_train_seq_path)[()]
            test_seq = np.load(fast_test_seq_path)
        else:
            train_data = np.load(fast_train_path)[()]
            test_data = np.load(fast_test_path)
    except:
        mfcc_train, mfcc_test = get_data_mfcc(data_dir, seq=False)
        fbank_train, fbank_test = get_data_fbank(data_dir, seq=False)

        train_x = np.column_stack((mfcc_train["x"], fbank_train["x"]))

        train_data = {"x": train_x, "y": mfcc_train["y"]}
        test_data = np.column_stack((mfcc_test, fbank_test))

        mfcc_train_seq, mfcc_test_seq = get_data_mfcc(data_dir, seq=True)
        fbank_train_seq, fbank_test_seq = get_data_fbank(data_dir, seq=True)

        train_seq_x = [np.column_stack((mfcc_train_seq["x"][i], fbank_train_seq["x"][i])) for i in
                       range(len(mfcc_train_seq["x"]))]

        train_seq_stack = {"x": train_seq_x, "y": mfcc_train_seq["y"]}
        test_seq = [np.column_stack((mfcc_test_seq[i], fbank_test_seq[i])) for i in range(len(mfcc_test_seq))]

        np.save(fast_train_path, train_data)
        np.save(fast_test_path, test_data)

        np.save(fast_train_seq_path, train_seq_stack)
        np.save(fast_test_seq_path, test_seq)

    if seq:
        return train_seq_stack, test_seq
    else:
        return train_data, test_data
