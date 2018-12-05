# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-1 下午7:56
# @file  : batcher.py
import os
import numpy as np
import pandas as pd
import random
from example import Example


class Batch(object):
    """
    Parameters:
        video_batch: np.ndarray shape=(batch_size, 80, 4096)
        caption_batch: np.ndarray shape=(batch_size, 20)
        video_mask: np.ndarray shape=(batch_size, 80)
        caption_mask: np.ndarray shape=(batch_size, 20)
    """
    def __init__(self, examplelist, train_mode=True):
        video_batch = []
        video_mask = []
        caption_batch_input = []
        caption_batch_target =[]
        caption_mask = []
        for idx, example in enumerate(examplelist):
            video_batch.append(np.expand_dims(example.video.video_features, axis=0))
            video_mask.append(np.expand_dims(example.video_mask, axis=0))
            if train_mode:
                caption_batch_input.append(np.expand_dims(example.caption.caption_input, axis=0))
                caption_batch_target.append(np.expand_dims(example.caption.caption_target, axis=0))
                caption_mask.append(np.expand_dims(example.caption_mask, axis=0))

        self.video_batch = np.concatenate(video_batch, axis=0)
        self.video_mask = np.concatenate(video_mask, axis=0)
        if train_mode:
            # batch_size * max_dec_len
            self.caption_batch_input = np.concatenate(caption_batch_input, axis=0)
            self.caption_batch_target = np.concatenate(caption_batch_target, axis=0)
            self.caption_mask = np.concatenate(caption_mask, axis=0)

    def make_dict(self,
                  video_placeholder=None,
                  caption_input_placeholder=None,
                  caption_target_placeholder=None,
                  video_mask=None,
                  caption_mask=None,
                  mean=None,
                  var=None):
        """Make a feed dict to feed into the model

        # Arguments:
            video_placeholder:
            caption_placeholder:
        """
        assert mean is not None and var is not None
        feed_dict = {}
        if video_placeholder is not None:
            feed_dict[video_placeholder] = (self.video_batch - mean) / var
        if caption_input_placeholder is not None:
            feed_dict[caption_input_placeholder] = self.caption_batch_input
        if caption_target_placeholder is not None:
            feed_dict[caption_target_placeholder] = self.caption_batch_target
        if video_mask is not None:
            feed_dict[video_mask] = self.video_mask
        if  caption_mask is not None:
            feed_dict[caption_mask] = self.caption_mask

        return feed_dict


class Batcher(object):
    """Batcher is a data loader, this object can
    return batch continuously

    """
    def __init__(self, hparams, single_pass=False):
        self.hparams = hparams
        self.csv_file = hparams.csv_file
        self.video_data = self.get_video_data()
        self.single_pass = single_pass

    """
    def get_video_train_data(self, relative_path):
        video_data = pd.read_csv(self.csv_file)
        video_data = video_data[video_data["Language"] == "English"]
        video_data["video_path"] = video_data.apply(lambda row: row["VideoID"]+"_"+str(int(row["Start"]))+"_"+str(int(row["End"]))+".avi", axis=1)
        video_data["video_path"] = video_data["video_path"].map(lambda x: os.path.join(relative_path, x))
        video_data = video_data[video_data["video_path"].map(lambda x: os.path.exists(x))]
        video_data = video_data[video_data["Description"].map(lambda x: isinstance(x, str))]
        unique_filenames = sorted(video_data["video_path"].unique())
        train_data = video_data[video_data["video_path"].map(lambda x: x in unique_filenames)]
        return train_data
    """

    def get_video_data(self):
        video_data = pd.read_csv(self.csv_file)
        return video_data

    def get_batch_generator(self, mode="train", single_pass=False):
        """Get a generator which is to yield a Batch Instance

        mode: can be train, eval, infer
        single_pass: if True, then this
        """

        assert mode in ["train", "eval", "infer"], "model can be {train, eval, infer}"
        mode_data = self.video_data[self.video_data["mode"] == mode]
        video_captions = zip(mode_data["video_path"].values, mode_data["caption"].values)
        example_num = len(video_captions)
        print "mode = {mode} and the sample num is {sample_num}".format(mode=mode, sample_num=example_num)
        while True:
            random.shuffle(video_captions)
            for start, end in zip(
                    range(0, example_num, self.hparams.batch_size),
                    range(self.hparams.batch_size, example_num, self.hparams.batch_size)):
                example_list = video_captions[start: end]

                def _load_video_feat(video_path):
                    return np.load(video_path)
                example_list = map(lambda example: Example(_load_video_feat(example[0]), example[1],
                                                           vocab=self.hparams.word2id), example_list)
                yield Batch(example_list)

            if self.single_pass or single_pass:
                print "infer mode: no more data"
                break
