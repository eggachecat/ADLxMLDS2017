import pandas as pd
import numpy as np
import sys
import os

import itertools


def get_phone_idx_map(data_dir="./data"):
    _48phone_idx_map = pd.read_csv(os.path.join(data_dir, "48phone_char.map"), header=None, delimiter="\t")
    _48phone_idx_map.set_index(0, drop=True, inplace=True)
    phone_idx_map = _48phone_idx_map[[1]].to_dict()[1]
    phone_char_map = _48phone_idx_map[[2]].to_dict()[2]

    return phone_idx_map, phone_char_map


def get_raw_label(data_dir="./data"):
    label_dir = os.path.join(data_dir, "label")
    label_path = os.path.join(label_dir, "train.lab")
    return pd.read_csv(label_path, header=None, delimiter=",")


def get_idx_phone_map(data_dir="./data"):
    _48_39map = pd.read_csv(os.path.join(data_dir, "phones/48_39.map"), header=None, delimiter="\t")
    _48_39map.set_index(0, drop=True, inplace=True)
    _48_39map = _48_39map[[1]].to_dict()[1]

    phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)
    idx_phone_map = dict([(phone_idx_map[k], _48_39map[k]) for k in phone_idx_map])

    return idx_phone_map, phone_char_map


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
        s += 1

    e = len(phone_list) - 1
    while phone_list[e] == sli:
        e -= 1

    phone_list = phone_list[s:e + 1]
    return [k for k, g in itertools.groupby(phone_list)]


def get_true_str():
    df = get_raw_label()
    idx_phone_map, phone_char_map = get_idx_phone_map()

    _48_39map = pd.read_csv(os.path.join("./data", "phones/48_39.map"), header=None, delimiter="\t")
    _48_39map.set_index(0, drop=True, inplace=True)
    _48_39map = _48_39map[[1]].to_dict()[1]

    _uid, _sid, _fid = df[0].str.split('_').str
    df["iid"], df["ts"] = _uid + "_" + _sid, _fid.apply(int)
    split_indices = df.groupby("iid").ts.count()
    idx = df["iid"].unique()
    split_indices = np.cumsum(split_indices[idx].as_matrix())

    seq = np.split(df[df.columns[1]].as_matrix(), split_indices)[:-1]

    ans_seq = []
    for _seq in seq:
        __seq = "".join(trim([phone_char_map[_48_39map[s]] for s in _seq]))
        ans_seq.append(__seq)

    ans = pd.DataFrame({"id": idx, "ans": ans_seq})
    ans.to_csv("./data/ans.csv")


def main(*args):
    target_path = args[0][1]
    if len(target_path[0]) > 1:
        threshold = args[0][2]
    else:
        threshold = -1
    threshold = float(threshold)

    ans = pd.read_csv("./data/ans.csv")
    test = pd.read_csv(target_path)
    merged = pd.merge(ans, test, left_on="id", right_on="id")
    tuples = [tuple(x) for x in merged[["ans", "phone_sequence"]].values]

    distance_ctr = 0
    per_ctr = 0
    ctr = 0

    for i, val in enumerate(tuples):
        if i > threshold * len(tuples):
            ans = val[0]
            predict = val[1]
            distance = edit_distance(ans, predict)
            distance_ctr += distance
            per_ctr += distance / len(ans)
            ctr += 1

    print("Avg edit distance: ", distance_ctr / ctr)
    print("Avg PER: ", per_ctr / ctr)


if __name__ == '__main__':
    # get_true_str()


    # print(pd.merge(ans, _ans, left_on="id", right_on="id"))
    main(sys.argv)
    # rng = check_random_state(random_state)
