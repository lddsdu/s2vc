# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-1 下午3:15
# @file  : main.py

from model import Model
import tensorflow as tf
from object.batcher import Batcher
from hparams import HParams
import random
import numpy as np


def main():
    hparams = HParams()
    batcher = Batcher(hparams)
    train_data_generator = batcher.get_batch_generator(mode="train")

    model = Model(hparams)

    # placeholder
    video = model.video  # batch_size, 30, 4096
    caption_input = model.caption_input  # batch_size, 20
    caption_target = model.caption_target  # batch_size, 20
    video_mask = model.video_mask  # batch_size, 80
    caption_mask = model.caption_mask  # batch_size, 20

    # loss & probs
    loss, probs = model.loss, model.probs  # scalar | batch_size, dec_max_len

    # summary
    loss_sum = tf.summary.scalar("loss", loss)
    summary_writer = tf.summary.FileWriter(hparams.summary_path, graph=tf.get_default_graph())

    # 初始化值为0.
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # initializer & session
    trainable_variables = tf.trainable_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # train & eval
    gradients = tf.gradients(loss, trainable_variables)
    grads, global_normal = tf.clip_by_global_norm(gradients, hparams.max_grad_norm)
    dynamic_learning_rate = tf.train.exponential_decay(hparams.learning_rate,
                                                       global_step,
                                                       hparams.decay_interval,
                                                       hparams.learning_decay,
                                                       staircase=False)
    optimizer = tf.train.AdagradOptimizer(dynamic_learning_rate)

    # apply_gradients(zip(grads, trainable_variables)
    train_op = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step, name="train_op")

    initialzer = tf.global_variables_initializer()
    session.run(initialzer)

    saver = tf.train.Saver()

    check_point_file = "/dl_data/video_caption/s2vc-ckpt-25000"
    print "restore weight from {}".format(check_point_file)
    saver.restore(session, check_point_file)

    for i in xrange(hparams.train_steps):
        # train
        batch = train_data_generator.next()
        feed_dict = batch.make_dict(video_placeholder=video,
                                    caption_input_placeholder=caption_input,
                                    caption_target_placeholder=caption_target,
                                    video_mask=video_mask,
                                    caption_mask=caption_mask,
                                    mean=hparams.mean,
                                    var=hparams.var)
        dl_rate, _, ls, ls_sum, _ = session.run([dynamic_learning_rate, train_op, loss, loss_sum, probs],
                                                feed_dict=feed_dict)
        print "[{}] lr_rate:{} loss:{}".format(i, dl_rate, ls)
        summary_writer.add_summary(ls_sum)

        # eval
        if i != 0 and i % hparams.eval_interval == 0:
            losses = []
            eval_batch_generator = batcher.get_batch_generator(mode="eval", single_pass=True)
            for eval_batch in eval_batch_generator:
                feed_dict = eval_batch.make_dict(video_placeholder=video,
                                                 caption_input_placeholder=caption_input,
                                                 caption_target_placeholder=caption_target,
                                                 video_mask=video_mask,
                                                 caption_mask=caption_mask,
                                                 mean=hparams.mean,
                                                 var=hparams.var)
                eval_ls, eval_probs = session.run([loss, probs], feed_dict=feed_dict)

                losses.append(eval_ls)
                randix = random.randint(0, len(eval_probs) - 1)         # randint 包含了up limit
                print "eval_loss: {eval_loss}".format(eval_loss=eval_ls)

                problist = np.argmax(eval_probs[randix, :, :], axis=0).tolist()
                eval_output = " ".join(map(lambda x: hparams.id2word[x], problist))

                targlist = eval_batch.caption_batch_target[randix, :].tolist()
                eval_target = " ".join(map(lambda x: hparams.id2word[x], targlist))

                print "output: {}\ntarget: {}".format(eval_output, eval_target)
            print "eval_avg_loss: {eval_avg_loss}".format(eval_avg_loss=np.mean(losses))

        if i != 0 and i % hparams.save_interval == 0:
            saver.save(session, "/dl_data/video_caption/s2vc-ckpt-{}".format(i))


if __name__ == '__main__':
    main()
