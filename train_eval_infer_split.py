# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-2 下午7:49
# @file  : train_eval_infer_split.py

"""this code is used to split train, eval, infer dataset,
and write necessary data into .csv file.
"""
import os
import pandas as pd
import random
train, eeval, infer = 8, 1, 1


def main():
    sumall = train + eeval + infer
    sumall = sumall * 1.0
    train_rate, eval_rate, infer_rate = train / sumall, eeval /sumall, infer / sumall

    video_data = pd.read_csv("video_corpus.csv")
    video_data = video_data[video_data["Language"] == "English"]
    video_data["video_path"] = video_data.apply(lambda row: row["VideoID"] + "_" + str(int(row["Start"])) + "_" + str(int(row["End"])) + ".avi", axis=1)
    video_data["video_path"] = video_data["video_path"].map(lambda x: os.path.join("data_preprocess/vgg/train_data_30", x))
    video_data = video_data[video_data["video_path"].map(lambda x: os.path.exists(x))]
    video_data = video_data[video_data["Description"].map(lambda x: isinstance(x, str))]
    unique_filenames = sorted(video_data["video_path"].unique())
    all_data = video_data[video_data["video_path"].map(lambda x: x in unique_filenames)]

    video_paths = all_data["video_path"].values
    descriptions = all_data["Description"].values

    video_captions = zip(video_paths, descriptions)
    random.shuffle(video_captions)
    all_data_num = len(video_captions)
    print "size {}".format(all_data_num)
    split_index1 = int(all_data_num * train_rate)
    split_index2 = int(all_data_num * (train_rate + eval_rate))
    train_data = video_captions[:split_index1]
    eval_data = video_captions[split_index1: split_index2]
    infer_data = video_captions[split_index2:]

    # save these into csv file
    # title: video_path, caption, mode
    train_df = pd.concat([pd.DataFrame(data=[[sample[0], sample[1], "train"]], columns=["video_path", "caption", "mode"]) for sample in train_data])
    eval_df = pd.concat([pd.DataFrame(data=[[sample[0], sample[1], "eval"]], columns=["video_path", "caption", "mode"]) for sample in eval_data])
    infer_df = pd.concat([pd.DataFrame(data=[[sample[0], sample[1], "infer"]], columns=["video_path", "caption", "mode"]) for sample in infer_data])
    all_df = pd.concat([train_df, eval_df, infer_df])
    all_df.to_csv("video_processed.csv")


if __name__ == '__main__':
    main()