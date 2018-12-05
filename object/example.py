# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-1 下午7:56
# @file  : example.py

import numpy as np


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"


class Example(object):
    """
    caption: Caption instance
    video: Video instance
    video_mask: ndarray shape=(video_max_len, ) dtype=np.float32
    caption_mask: ndarray shape=(caption_max_len, ) dtype=np.float32
    """
    def __init__(self, video_features, description, vocab=None, video_max_len=30, caption_max_len=20):
        """创建example 实例

        :param video_features: a ndarray shaped of 80, 4096
        :param description: a string
        :param video_max_len: 80
        :param caption_max_len: 20
        """
        self.vocab = vocab
        self.caption = Caption(description, caption_max_len, vocab=vocab)
        self.video = Video(video_features, video_max_len)

        # video mask
        video_mask = np.ones(shape=(self.video.valid_len, ), dtype=np.float32)
        self.video_mask = video_mask if not self.video.need_mask() \
            else np.concatenate((video_mask,
            np.zeros(video_max_len - self.video.valid_len, dtype=np.float32)), axis=0)

        # caption mask
        caption_mask = np.ones(shape=(self.caption.valid_len, ), dtype=np.float32)
        self.caption_mask = caption_mask if not self.caption.need_mask() \
            else np.concatenate((caption_mask,
            np.zeros(self.caption.max_len - self.caption.valid_len, dtype=np.float32)), axis=0)
        pass


class Caption(object):
    """
    Parameters:
        description: original string description
        max_len: 20
        caption: list of integer
        valid_len: the valid length of caption
    """
    def __init__(self, description, max_len, vocab=None):
        assert isinstance(vocab, dict), "vocab should be a dictionary"
        self.description = description
        self.max_len = max_len
        self.caption_input = None
        self.caption_target = None
        self.valid_len = 0
        self.format(vocab)

    def format(self, vocab):

        # input
        temp = self.description
        temp = temp.strip().split()
        temp = self.replace_dot(temp)
        temp = map(lambda x: vocab[x] if x in vocab else vocab[UNK], temp)

        # caption input like <s> xx xx xx </s> </s>
        caption_input =[vocab[SOS]] + temp
        if len(caption_input) > self.max_len:
            caption_input = caption_input[:self.max_len]
        caption_input += [vocab[EOS]] * (self.max_len - len(caption_input))
        self.caption_input = caption_input

        # caption target like xx xx xx </s> </s> </s>
        caption_target = temp
        if len(caption_target) >= self.max_len:
            caption_target = caption_target[:self.max_len - 1]
        self.valid_len = len(caption_target) + 1
        while len(caption_target) < self.max_len:
            caption_target += [vocab[EOS]]
        self.caption_target = caption_target

    def replace_dot(self, captions):
        captions = map(lambda x: x.replace('.', ''), captions)
        captions = map(lambda x: x.replace(',', ''), captions)
        captions = map(lambda x: x.replace('"', ''), captions)
        captions = map(lambda x: x.replace('\n', ''), captions)
        captions = map(lambda x: x.replace('?', ''), captions)
        captions = map(lambda x: x.replace('!', ''), captions)
        captions = map(lambda x: x.replace('\\', ''), captions)
        captions = map(lambda x: x.replace('/', ''), captions)
        return captions

    def need_mask(self):
        return self.valid_len < self.max_len


class Video(object):
    """
    Parameters:
        original_video_features: a ndarray shaped valid_len(80), 4096
        max_len: 80
        valid_len: the original_video_features's first dimension size
        video_features: a ndarray 80, 4096
    """
    def __init__(self, video_features, max_len):
        self.original_video_features = video_features
        self.max_len = max_len
        self.valid_len = 0
        self.video_features = None
        self.reformat()
        pass

    def reformat(self):
        self.valid_len = self.original_video_features.shape[0]
        if self.valid_len == self.max_len:
            self.video_features = self.original_video_features
        else:
            image_feature_shape = list(self.original_video_features.shape[1:])
            padding = np.zeros([self.max_len - self.valid_len] + image_feature_shape, dtype=np.float32)
            self.video_features = np.concatenate((self.original_video_features, padding), axis=0)

    def need_mask(self):
        return self.valid_len < self.max_len