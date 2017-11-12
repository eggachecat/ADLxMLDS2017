import json
import os
import numpy as np

import re


class DataUtils:
    def __init__(self, root_data_path):
        self.root_data_path = root_data_path

    @staticmethod
    def load_label(file_path):
        with open(file_path) as fp:
            data_list = json.load(fp)

        return data_list

    @staticmethod
    def format_caption(caption):
        caption = caption.replace(",", "").replace(".", "").replace(":", "").replace("!", "").replace("  ", "").replace(
            "'s", "is").replace("'", "").replace("&", "").replace("(", "").replace(")", "").replace("[", "").replace(
            "]", "").replace("\"", "")
        return '<BOS> ' + caption.strip().lower() + ' <EOS>'

    @staticmethod
    def get_words(caption):
        words = caption.split(" ")
        return words

    @staticmethod
    def get_dictionary(id_caption_obj=None):
        if id_caption_obj is None:
            try:
                w2i = json.load(open("./data/w2i_dict.json"))
            except FileNotFoundError:
                print("need dictionary !!")
                exit()
        else:
            try:
                w2i = json.load(open("./data/w2i_dict.json"))
            except FileNotFoundError:
                all_words_set = {"<BOS>", "<EOS>", "<UNK>", "<PAD>"}
                for _id in id_caption_obj:
                    words_set = set()
                    for words in id_caption_obj[_id]:
                        words_set = words_set | set(words)
                    all_words_set = all_words_set | words_set

                all_words_set = list(all_words_set)
                all_words_set = [word for word in all_words_set if
                                 (not bool(re.search(r'\d', word)) and (word is not ''))]
                all_words_set.sort()
                w2i = dict([(word, i) for i, word in enumerate(all_words_set)])
                try:
                    with open("./data/w2i_dict.json", "w") as fp:
                        json.dump(w2i, fp)
                except IOError:
                    print("failed to save")

        i2w = list(w2i.keys())
        i2w.sort()
        return w2i, i2w

    @staticmethod
    def str_to_label(id_caption_obj, w2i):
        for _id in id_caption_obj:
            id_caption_obj[_id] = [[w2i[word] if word in w2i else w2i["<UNK>"] for word in captaion] for
                                   captaion in id_caption_obj[_id]]
        return id_caption_obj

    def get_id_caption_obj(self, label_file_name):
        file_name = os.path.join(self.root_data_path, label_file_name)
        data_list = self.load_label(file_name)
        return dict(
            [(v["id"], [self.get_words(self.format_caption(caption)) for caption in v["caption"]]) for v in data_list])

    def get_id_label_obj(self, label_file_name):
        id_caption_obj = self.get_id_caption_obj(label_file_name)
        w2i, iw2 = self.get_dictionary(id_caption_obj)
        return self.str_to_label(id_caption_obj, w2i)

    def load_feat(self, id, d_type="train"):
        if d_type == "train":
            return np.load(os.path.join(self.root_data_path, "training_data/feat/{}.npy".format(id)))
        else:
            return np.load(os.path.join(self.root_data_path, "testing_data/feat/{}.npy".format(id)))

    def load_source_dataset(self, ordered_id_list=None):
        if ordered_id_list is None:
            try:
                source_dataset = np.load("./data/source_dataset.npy")
            except FileNotFoundError:
                print("need ordered_id_list !!")
                exit()
        else:
            try:
                source_dataset = np.load("./data/source_dataset.npy")
            except FileNotFoundError:
                source_dataset = []
                for id in ordered_id_list:
                    print("loading {}".format(id))
                    source_dataset.append(self.load_feat(id))

                source_dataset = np.array(source_dataset)

                try:
                    np.save("./data/source_dataset.npy", source_dataset)
                except IOError:
                    print("failed to save")

        return source_dataset

    def batch_generator(self, batch_size):
        id_label_obj = self.get_id_label_obj("training_label.json")

        # make sure always keys in the same order
        ordered_id_list = list(id_label_obj.keys())
        ordered_id_list.sort()

        target_dataset = np.array([id_label_obj[id][0] for id in ordered_id_list])

        max_len = max([len(t) for t in target_dataset])
        target_dataset_mask = np.array(
            [np.concatenate((np.ones(len(t)), np.zeros(max_len - len(t)))) for t in target_dataset])
        target_dataset = np.array([np.pad(t, (0, max_len - len(t)), "edge") for t in target_dataset])
        source_dataset = self.load_source_dataset(ordered_id_list)

        assert target_dataset.shape[0] == source_dataset.shape[0]

        sample_len = target_dataset.shape[0]
        print(sample_len)
        n_batch = sample_len // batch_size
        indices_pool = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batch)]

        while True:
            indices_order = np.random.permutation(n_batch)
            for index in indices_order:
                split_pair = indices_pool[index]
                s, e = split_pair[0], split_pair[1]
                yield source_dataset[s:e], target_dataset[s:e], target_dataset_mask[s:e]


def demo(batch_size):
    sample_len = 100
    n_batch = sample_len // batch_size
    indices_pool = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batch)]

    while True:
        indices_order = np.random.permutation(n_batch)
        for index in indices_order:
            split_pair = indices_pool[index]
            s, e = split_pair[0], split_pair[1]
            yield list(range(s, e))


def main(root_data_path):
    du = DataUtils(root_data_path)
    id_caption_obj = du.get_id_caption_obj("training_label.json")
    w2i, i2w = du.get_dictionary(id_caption_obj)

    print(i2w)
    batch_generator = du.batch_generator(10)

    encoder_inputs, decoder_inputs, decoder_inputs_mask = batch_generator.__next__()
    print(decoder_inputs[0:1])
    print(decoder_inputs_mask[0:1])
    print(decoder_inputs[0].shape)
    print(decoder_inputs[1].shape)


if __name__ == '__main__':
    main("D:\\workstation\\adl\\data\\hw2")
