import time
import tensorflow as tf
import numpy as np
import skimage
import skimage.io
import skimage.transform
import scipy.misc
from skimage.viewer import ImageViewer
import pylab as plt
import json
from sklearn.preprocessing import LabelBinarizer

GLOBAL_HAIRS = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
GLOBAL_EYES = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']

HAIR_ENCODER = LabelBinarizer()
HAIR_ENCODER.fit(GLOBAL_HAIRS)

EYES_ENCODER = LabelBinarizer()
EYES_ENCODER.fit(GLOBAL_EYES)

HAIR_CODES = HAIR_ENCODER.transform(GLOBAL_HAIRS)
EYES_CODES = EYES_ENCODER.transform(GLOBAL_EYES)
ALL_CONDITIONS = []
for h in HAIR_CODES:
    for e in EYES_CODES:
        ALL_CONDITIONS.append(np.concatenate((h, e)))

ALL_CONDITIONS = np.array(ALL_CONDITIONS)


def preprocessing(data_path, use_augment=True, just_id=True):
    print("start loading....")
    with open("{}/tags_clean.csv".format(data_path), "r") as fp:
        lines = fp.readlines()

    hairs = [hair + " hair" for hair in GLOBAL_HAIRS]
    eyes = [eye + " eyes" for eye in GLOBAL_EYES]

    info_obj = dict()
    for line in lines:
        id_, content = line.replace("\n", "").split(",")
        info = dict()
        for tag in content.split("\t")[:-1]:
            keyword = tag.split(":")[0].strip()
            if keyword in hairs:
                info["hair"] = keyword.split(" ")[0]
            if keyword in eyes:
                info["eyes"] = keyword.split(" ")[0]

        if "hair" in info and "eyes" in info:
            hair_code = HAIR_ENCODER.transform([info["hair"]])
            eyes_code = EYES_ENCODER.transform([info["eyes"]])
            info["encode"] = np.concatenate((hair_code, eyes_code), axis=1)
            info_obj[id_] = info

    for id_ in frozenset(info_obj.keys()):
        if not just_id:
            img = skimage.io.imread("{}/faces/{}.jpg".format(data_path, id_))
            img_resized = skimage.transform.resize(img, (64, 64), mode='constant')
            scipy.misc.imsave("{}/faces_resized/{}.jpg".format(data_path, id_), img_resized)

        if use_augment:
            for i, angle in enumerate([-20, -10, 0, 10, 20]):
                for flip in [0, 1]:
                    if just_id:
                        id__ = "{}-{}-{}".format(id_, i, flip)
                        info_obj[id__] = info_obj[id_]
                    else:
                        img_rotated = skimage.transform.rotate(img_resized, angle, mode='edge')
                        if flip:
                            img_new = np.fliplr(img_rotated)
                        else:
                            img_new = img_rotated
                        scipy.misc.imsave("{}/faces_resized/{}.jpg".format(data_path, id__), img_new)

    np.save("./data/info_obj.npy", info_obj)


def read_data_():
    info_obj = np.load('./data/info_obj.npy').item()
    img_obj = dict()
    ctr = 0
    start = time.time()
    for id_ in info_obj:
        ctr += 1
        print(id_)
        img_obj[id_] = skimage.io.imread("./data/faces_resized/{}.jpg".format(id_))
        if ctr % 1510 == 0:
            print(ctr / 1510)
    print("Loading image with {}s".format(time.time() - start))

    return info_obj, img_obj


import pylab as plt


def read_data():
    info_obj = np.load('./data/info_obj.npy').item()

    start = time.time()
    ids = list(info_obj.keys())
    images = skimage.io.imread_collection(["./data/faces_resized/{}.jpg".format(id_) for id_ in ids])
    print(len(images))
    print("Loading image with {}s".format(time.time() - start))
    #
    # img_obj = dict()
    # start = time.time()
    # for i, id_ in enumerate(ids):
    #     img_obj[id_] = images[i]
    #     if i % 1000 == 0:
    #         print(i)
    # print("Making dict with {}s".format(time.time() - start))

    return info_obj, images


def read_data_obj():
    info_obj = np.load('./data/info_obj.npy').item()

    start = time.time()
    ids = list(info_obj.keys())
    images = skimage.io.imread_collection(["./data/faces_resized/{}.jpg".format(id_) for id_ in ids])
    print(len(images))
    print("Loading image with {}s".format(time.time() - start))

    img_obj = dict()
    start = time.time()
    for i, id_ in enumerate(ids):
        img_obj[id_] = images[i]
        if i % 1000 == 0:
            print(i)
    print("Making dict with {}s".format(time.time() - start))
    return info_obj, img_obj


def read_data_debug():
    info_obj = np.load('./data/info_obj.npy').item()
    img_obj = dict()
    const_ids = ["2", "17"]
    ids = []
    for id_ in const_ids:
        ids.append(id_)
        for i in range(5):
            for j in [0, 1]:
                ids.append("{}-{}-{}".format(id_, i, j))
    info_obj_ = dict([(id_, info_obj[id_]) for id_ in ids])
    images = skimage.io.imread_collection(["./data/faces_resized/{}.jpg".format(id_) for id_ in ids])

    return info_obj_, images


def get_test_conditions():
    return np.array(ALL_CONDITIONS)


def save_samples(samples, test_path, hp):
    for i in range(len(GLOBAL_HAIRS)):
        for j in range(len(GLOBAL_EYES)):
            image_index = i * len(GLOBAL_EYES) + j
            image = samples[image_index]

            scipy.misc.imsave(
                "{}/{}-hair_{}-eyes.jpg".format(test_path, GLOBAL_HAIRS[i], GLOBAL_EYES[j]), image)


import csv


def read_infer_data(data_path):
    conditions = []
    text_ids = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        for text_id, text in reader:
            text_id = int(text_id)
            text_list = text.lower().split(" ")

            if len(text_list) > 2:
                if text_list[1] == "hair":
                    hair_code = HAIR_ENCODER.transform([text_list[0]])
                    eyes_code = EYES_ENCODER.transform([text_list[2]])
                    condition = np.concatenate((hair_code, eyes_code), axis=1)[0]
                else:
                    hair_code = HAIR_ENCODER.transform([text_list[2]])
                    eyes_code = EYES_ENCODER.transform([text_list[0]])
                    condition = np.concatenate((hair_code, eyes_code), axis=1)[0]
            else:
                if text_list[1] == "hair":
                    hair_code = HAIR_ENCODER.transform([text_list[0]])
                    eyes_code = EYES_ENCODER.transform(np.random.choice(GLOBAL_EYES, 1))
                    condition = np.concatenate((hair_code, eyes_code), axis=1)[0]
                else:
                    hair_code = HAIR_ENCODER.transform(np.random.choice(GLOBAL_HAIRS, 1))
                    eyes_code = EYES_ENCODER.transform([text_list[0]])
                    condition = np.concatenate((hair_code, eyes_code), axis=1)[0]

            conditions.append(condition)
            text_ids.append(text_id)

    return np.array(text_ids), np.array(conditions)


def main():
    info_obj = np.load('./data/info_obj.npy').item()
    img_obj = dict()
    ctr = 0
    start = time.time()

    print(np.array([info_obj[id_]["encode"][0] for id_ in list(info_obj.keys())[:2]]).shape)

    for id_ in info_obj:
        print(info_obj[id_]["encode"][0].shape)
        exit()
        ctr += 1
        img_obj[id_] = skimage.io.imread("./data/faces_resized/{}.jpg".format(id_))
        if ctr % 1510 == 0:
            print(ctr / 1510)
    print("Loading image with {}s".format(time.time() - start))


if __name__ == '__main__':
    preprocessing("./data/")
    # data_path = "./test_tag.txt"
    # text_ids, conditions = read_infer_data(data_path)
    # print(conditions.shape, conditions)
