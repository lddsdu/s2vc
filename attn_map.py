# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-5 下午3:14
# @file  : attn_map.py

import matplotlib.pyplot as plt
idx = 0


def attention_map(activation_map, x_ticks=None, y_ticks=None, store_path=None):
    """This method is used to plot heat map

    :param activation_map: a 2d array, first dimension
    :param x_ticks: the output
    :param y_ticks: the input
    :param store_path: string, to store a the pdf file
    """
    shape = activation_map.shape
    assert len(shape) == 2, "attention_map should be a 2D array"
    if x_ticks is None:
        x_ticks = map(str, range(shape[1]))
    assert y_ticks is not None, "y_tricks is the label of output"
    global idx
    idx += 1
    # plt.clf()
    if idx % 100 == 0:
        plt.close("all")
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)
    i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Probability', labelpad=2)

    ax.set_yticks(range(shape[0]))
    ax.set_yticklabels(y_ticks)
    ax.set_xticks(range(shape[1]))
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    ax.grid()
    f.savefig(store_path)
