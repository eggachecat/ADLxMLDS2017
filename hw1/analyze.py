import matplotlib.pyplot as plt
import numpy as np


def mannual_ensumble(ctr_path_list):
    if type(ctr_path_list) is not list:
        ctr_obj_list = [ctr_path_list]

    ctr_obj_list = [np.load(p) for p in ctr_path_list]
    x_label = np.arange(48)
    color_list = ["r", "b"]
    opacity = 0.4
    bar_width = 0.35

    for i, ctr_dict in enumerate(ctr_obj_list):

        y_data = []
        ctr_dict = ctr_dict[()]

        for x in range(48):
            y_data.append(ctr_dict[x]["true"] / ctr_dict[x]["freq"])

        plt.bar(x_label + bar_width * i, y_data, bar_width,
                alpha=opacity,
                color=color_list[i],
                label=ctr_path_list[i])

    plt.xticks(x_label + bar_width / 2, tuple(x_label))
    plt.legend()

    plt.tight_layout()
    plt.show()


mannual_ensumble(["1508951634.1246283.npy", "1508951683.6492145.npy"])
