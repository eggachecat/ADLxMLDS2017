import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import keras
import time
import itertools

from keras.preprocessing import sequence


def get_phone_idx_map(data_dir="./data/"):
    _48phone_idx_map = pd.read_csv(os.path.join(data_dir, "48phone_char.map"), header=None, delimiter="\t")
    _48phone_idx_map.set_index(0, drop=True, inplace=True)
    phone_idx_map = _48phone_idx_map[[1]].to_dict()[1]
    phone_char_map = _48phone_idx_map[[2]].to_dict()[2]

    return phone_idx_map, phone_char_map


def get_idx_phone_map(data_dir="./data/"):
    _48_39map = pd.read_csv(os.path.join(data_dir, "phones/48_39.map"), header=None, delimiter="\t")
    _48_39map.set_index(0, drop=True, inplace=True)
    _48_39map = _48_39map[[1]].to_dict()[1]

    phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)
    idx_phone_map = dict([(phone_idx_map[k], _48_39map[k]) for k in phone_idx_map])

    return idx_phone_map, phone_char_map


def get_raw_label(data_dir="./data/"):
    label_dir = os.path.join(data_dir, "label")
    label_path = os.path.join(label_dir, "train.lab")
    return pd.read_csv(label_path, header=None, delimiter=",")


def to_timestep_seq(df):
    _uid, _sid, _fid = df[0].str.split('_').str
    df["iid"], df["ts"] = _uid + "_" + _sid, _fid.apply(int)
    idx = df["iid"].unique()
    split_indices = df.groupby("iid").ts.count()
    split_indices = np.cumsum(split_indices[idx].as_matrix())

    seq = np.split(df[df.columns[1:-2]].as_matrix(), split_indices)
    return seq[:-1], idx


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


def super_trim(_str, body_thres=2, side_thres=5):
    _str_stat = [(k, sum(1 for _ in g)) for k, g in itertools.groupby(_str)]
    n_data = len(_str_stat)
    res_list = [_str_stat[0]]
    i = 1
    while i < n_data - 1:

        head = res_list.pop()
        body = _str_stat[i]
        tail = _str_stat[i + 1]

        if head[0] == tail[0]:
            if body[1] <= body_thres and (head[1] >= side_thres or tail[1] >= side_thres):
                res_list.extend([head, tail])
            else:
                res_list.extend([head, body, tail])
        else:
            if body[1] <= body_thres and (head[1] >= side_thres or tail[1] >= side_thres):
                res_list.extend([head, tail])
            else:
                res_list.extend([head, body, tail])

        i += 2

    # print(i, n_data)
    if i == n_data - 1:
        res_list.append(_str_stat[-1])

    # print(res_list)
    return list("".join([res[1] * res[0] for res in res_list]))


def trim(phone_list, sli="L", use_super_trim=None):
    s = 0
    if use_super_trim is not None:
        for i in range(use_super_trim):
            phone_list = super_trim(phone_list)

    while phone_list[s] == sli:
        s += 1

    e = len(phone_list) - 1
    while phone_list[e] == sli:
        e -= 1

    phone_list = phone_list[s:e + 1]
    return [k for k, g in itertools.groupby(phone_list)]


def _get_data(src_type, data_dir="./data/", seq=True):
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
            test_seq_stack = np.load(fast_test_seq_path)[()]
        else:
            train_data_stack = np.load(fast_train_path)[()]
            test_data_stack = np.load(fast_test_path)[()]
    except:

        phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)

        raw_train_data = pd.read_csv(train_path, header=None, delimiter=" ")
        train_label = get_raw_label(data_dir)
        train_label[1] = train_label[1].apply(lambda x: phone_idx_map[x])

        train_data = pd.merge(raw_train_data, train_label, left_on=0, right_on=0)

        train_data_stack = {"x": np.array(train_data[train_data.columns[1:-1]].as_matrix()),
                            "y": np.array(train_data[train_data.columns[-1]].as_matrix())}

        raw_test_data = pd.read_csv(test_path, header=None, delimiter=" ")
        test_data_stack = {"x": raw_test_data[raw_test_data.columns[1:]].as_matrix(),
                           "y": raw_test_data[raw_test_data.columns[0]].as_matrix()}

        if seq:
            raw_train_seq, _ = to_timestep_seq(train_data)
            raw_test_seq, idx = to_timestep_seq(raw_test_data)

            train_seq_stack = {"x": np.array([seq[:, :-1] for seq in raw_train_seq]),
                               "y": np.array([seq[:, -1] for seq in raw_train_seq])}

            test_seq_stack = {"x": np.array([seq for seq in raw_test_seq]),
                              "y": idx}
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


def get_data_mfcc(data_dir="./data/", seq=True):
    return _get_data("mfcc", data_dir, seq)


def get_data_fbank(data_dir="./data/", seq=True):
    return _get_data("fbank", data_dir, seq)


def get_data_full(data_dir="./data/", seq=True):
    fast_train_path = os.path.join(data_dir, "full_train.npy")
    fast_test_path = os.path.join(data_dir, "full_test.npy")

    fast_train_seq_path = os.path.join(data_dir, "full_train_seq.npy")
    fast_test_seq_path = os.path.join(data_dir, "full_test_seq.npy")

    try:
        if seq:
            train_seq_stack = np.load(fast_train_seq_path)[()]
            test_seq_stack = np.load(fast_test_seq_path)[()]
        else:
            train_data_stack = np.load(fast_train_path)[()]
            test_data_stack = np.load(fast_test_path)[()]
    except:
        mfcc_train, mfcc_test = get_data_mfcc(data_dir, seq=False)
        fbank_train, fbank_test = get_data_fbank(data_dir, seq=False)

        train_x = np.column_stack((mfcc_train["x"], fbank_train["x"]))
        test_x = np.column_stack((mfcc_test["x"], fbank_test["x"]))

        train_data_stack = {"x": train_x, "y": mfcc_train["y"]}
        test_data_stack = {"x": test_x, "y": mfcc_test["y"]}

        mfcc_train_seq, mfcc_test_seq = get_data_mfcc(data_dir, seq=True)
        fbank_train_seq, fbank_test_seq = get_data_fbank(data_dir, seq=True)

        train_seq_x = np.array([np.column_stack((mfcc_train_seq["x"][i], fbank_train_seq["x"][i])) for i in
                                range(len(mfcc_train_seq["x"]))])
        train_seq_stack = {"x": train_seq_x, "y": mfcc_train_seq["y"]}
        test_seq_stack = {
            "x": np.array(
                [np.column_stack((mfcc_test_seq["x"][i], fbank_test_seq["x"][i])) for i in
                 range(len(mfcc_test_seq["x"]))]),
            "y": mfcc_test_seq["y"]}

        np.save(fast_train_path, train_data_stack)
        np.save(fast_test_path, test_data_stack)

        np.save(fast_train_seq_path, train_seq_stack)
        np.save(fast_test_seq_path, test_seq_stack)

    if seq:
        return train_seq_stack, test_seq_stack
    else:
        return train_data_stack, test_data_stack


def get_train_idx(data_dir="./data/"):
    try:
        idx = np.load("./data//train_ans_idx.npy")
    except:
        df = pd.read_csv(data_dir + "/mfcc/train.ark", header=None, delimiter=" ")
        _uid, _sid, _fid = df[0].str.split('_').str
        df["iid"] = _uid + "_" + _sid
        idx = df["iid"].unique()
        np.save("./data//train_ans_idx.npy", idx)

    return idx


if __name__ == '__main__':
    print("get_train_idx")
    idx = get_train_idx()
    print(idx)
    print("=============")

    print("get_data_mfcc(seq=False)")
    train_data_stack, test_data_stack = get_data_mfcc(seq=False)
    print(train_data_stack["x"].shape, train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(test_data_stack["x"].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print("=============")

    print("get_data_fbank(seq=False)")
    train_data_stack, test_data_stack = get_data_fbank(seq=False)
    print(train_data_stack["x"].shape, train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(test_data_stack["x"].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print("=============")

    print("get_data_full(seq=False)")
    train_data_stack, test_data_stack = get_data_full(seq=False)
    print(train_data_stack["x"].shape, train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(test_data_stack["x"].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print("=============")

    print("get_data_mfcc(seq=True)")
    train_data_stack, test_data_stack = get_data_mfcc(seq=True)
    print(train_data_stack["x"].shape, len(train_data_stack["x"]), train_data_stack["x"][0].shape,
          train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, len(train_data_stack["y"]), train_data_stack["y"][0].shape,
          train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(len(test_data_stack["x"]), test_data_stack["x"][0].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print(len(test_data_stack["y"]), test_data_stack["y"][0], test_data_stack["y"][-1])
    print("-------------")
    print("=============")

    print("get_data_fbank(seq=True)")
    train_data_stack, test_data_stack = get_data_fbank(seq=True)
    print(train_data_stack["x"].shape, len(train_data_stack["x"]), train_data_stack["x"][0].shape,
          train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, len(train_data_stack["y"]), train_data_stack["y"][0].shape,
          train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(len(test_data_stack["x"]), test_data_stack["x"][0].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print(len(test_data_stack["y"]), test_data_stack["y"][0], test_data_stack["y"][-1])
    print("-------------")
    print("=============")

    print("get_data_full(seq=True)")
    train_data_stack, test_data_stack = get_data_full(seq=True)
    print(train_data_stack["x"].shape, len(train_data_stack["x"]), train_data_stack["x"][0].shape,
          train_data_stack["x"][0], train_data_stack["x"][-1])
    print("-------------")
    print(train_data_stack["y"].shape, len(train_data_stack["y"]), train_data_stack["y"][0].shape,
          train_data_stack["y"][0], train_data_stack["y"][-1])
    print("-------------")
    print(len(test_data_stack["x"]), test_data_stack["x"][0].shape, test_data_stack["x"][0], test_data_stack["x"][-1])
    print("-------------")
    print(len(test_data_stack["y"]), test_data_stack["y"][0], test_data_stack["y"][-1])
    print("-------------")
    print("=============")
