# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-5 上午8:28
# @file  : eval.py.py

import os
import numpy as np
import tensorflow as tf
import glob
from model import Model
from hparams import HParams
from attn_map import attention_map


def main():
    hparams = HParams()
    model = Model(hparams, mode="infer")

    # placeholder
    video = model.video
    video_mask = model.video_mask
    activation_map = model.activation_map       # dec_max_len, enc_max_len
    # probs.shape=(batch_size, time_step, vocab_size)
    probs = tf.transpose(model.probs, [0, 2, 1])

    print video, video_mask, probs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # saver
    saver = tf.train.Saver()
    print "reload weight from file"
    saver.restore(session, "/dl_data/video_caption/s2vc-ckpt-35000")

    for filename in glob.glob("/dl_data/video_caption/s2vc/data_preprocess/vgg/train_data_30/*.avi"):
        print filename
        video_data, video_mask_data = get_data(hparams, filename)
        heatmap_filename = filename[len(os.path.dirname(filename)) + 1:][:-4]
        prediction, activation_map_data = session.run([probs, activation_map],
                                                      feed_dict={video: video_data, video_mask: video_mask_data})
        prediction = prediction[0]
        prediction = np.argmax(prediction, axis=1)
        id2word = hparams.id2word
        p = []
        for item in prediction:
            p.append(id2word[int(item)])
        # activation_map_data.shape=(dec_max_len, enc_max_len)
        attention_map(activation_map_data, x_ticks=None, y_ticks=p, store_path=os.path.join("attention_maps", heatmap_filename))
        print " ".join(p)


def get_data(hparams, video_path):
    video_data = np.expand_dims(np.load(video_path), 0)
    video_data = (video_data - hparams.mean) / hparams.var
    video_mask_data = np.zeros((1, 30), np.float32)
    for time_index in range(video_data.shape[1]):
        video_mask_data[0, time_index] = 1.0
    return video_data, video_mask_data


if __name__ == '__main__':
    main()
