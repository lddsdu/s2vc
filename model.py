# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-1 下午3:16
# @file  : model.py


"""
This file is used to construct the model
"""
import time
import tensorflow as tf


class Model(object):
    """
    TODO: 扩展，在seq2seq添加infer的网络结构
    """
    def __init__(self, hparams, mode="train"):
        start = time.time()
        self.train_mode = False
        self.batch_size = None
        if mode == "train":
            self.train_mode = True
            self.batch_size = hparams.batch_size

        self.attention_list = []    # item shape=(batch_size, enc_max_len, 1)
        # infer 和 train mode下的decoder的情况不同
        self.infer_mode = not self.train_mode
        self.hparams = hparams
        if self.infer_mode:
            self.hparams.batch_size = 1

        self.video = self.caption_input = self.caption_target = self.video_mask = self.caption_mask = None
        self.encoder_outputs = None
        self.encoder_features = None
        self.add_placeholder()
        self.loss, self.probs = self.add_seq2seq()
        end = time.time()
        print "Construct computational graph consumes {t} s".format(t=(end - start))

        attentions = map(lambda x: tf.expand_dims(x, axis=0), self.attention_list)
        # dec_max_len, batch_size(1), enc_max_len, (1)
        self.activation_map = tf.squeeze(tf.concat(attentions, axis=0))

    def add_placeholder(self):
        """
        video, caption_input, caption_target, video_mask, caption_mask are all placeholder
        video: batch_size, enc_max_len, video_feature_dim
        caption_input: batch_size, dec_max_len
        caption_target: batch_size, dec_max_len
        video_mask: batch_size, enc_max_len
        caption_mask: batch_size, dec_max_len
        """
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.hparams.enc_max_len,
                                                 self.hparams.video_features_dim])
        self.caption_input = tf.placeholder(tf.int32, [self.batch_size, self.hparams.dec_max_len])
        self.caption_target = tf.placeholder(tf.int32, [self.batch_size, self.hparams.dec_max_len])
        # 在做 attention 的时候是必须的
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.hparams.enc_max_len])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.hparams.dec_max_len])

    def _compute_attention(self, state, scope_name="compute_attention", idx=0):
        """方法中还需要使用到
        self.encoder_outputs, a list(len = enc_max_len) of tensor(shape=[batch_size, hidden_units])
        self.video_mask: a tensor(shape= [batch_size, enc_max_len])
        state: batch_size * hidden_units

        e = softmax(v^T (W_h h_i + W_s s_t + b_attn ))

        W_h h_i :   batch_size, enc_max_len, hidden_units
        W_s s_t :   batch_size, 1, hidden_units
        v:          hidden_units,
        b_attn:     hidden_units,
        """
        with tf.variable_scope(scope_name):
            if idx > 0:
                tf.get_variable_scope().reuse_variables()

            W_s = tf.get_variable(name="W_s", shape=(self.hparams.hidden_units, self.hparams.hidden_units),
                                  dtype=tf.float32)
            b_attn = tf.get_variable(name="b_attn", shape=(self.hparams.hidden_units, ))
            decoder_feature = tf.matmul(self.reduce_state(state), W_s)     # batch_size, hidden_units
            decoder_feature = tf.expand_dims(decoder_feature, axis=1)   # batch_size, 1, hidden_units
            # broad cast add
            concat_features = self.encoder_features + decoder_feature   # batch_size, max_enc_len, hidden_units
            concat_features += b_attn   # batch_size, max_enc_len, hidden_units
            v = tf.get_variable(name="v", shape=(self.hparams.hidden_units, ), dtype=tf.float32)
            # batch_size, max_enc_len
            concat_features = tf.squeeze(tf.reduce_sum(tf.multiply(concat_features, v), axis=-1))

            # soft1.shape=(batch_size, max_enc_len)
            soft1 = tf.nn.softmax(concat_features, axis=-1) # softmax and then mask
            soft1 = soft1 * self.video_mask
            # soft2=(batch_size, max_enc_len, 1)
            soft2 = tf.expand_dims(soft1 / tf.reduce_sum(soft1, axis=-1), -1)
            if scope_name == "attention1":
                self.attention_list.append(soft2)
            # attn_vector shape=(batch_size, hidden_dims) |
            # encoder_outputs shape=(batch_size, max_enc_len, hidden_units)
            attn_vector = tf.reduce_sum(self.encoder_outputs * soft2, axis=1)
            return attn_vector

    def add_seq2seq(self):

        # 1, encoder
        with tf.variable_scope("encoder"):
            """
            github中的代码在encoder, decoder部分使用的lstm cell是相同的， 在原始的输入的时候维度
            也需要相同，所以加了padding, tf.concat((image_features, padding), 1)确保了输入的维度
            为batch_size, 2 * hidden_units
            """
            # image feature scale weights and bias
            encoder_image_W = tf.Variable(
                tf.truncated_normal([self.hparams.video_features_dim, self.hparams.hidden_units]),
                name="encoder_image_W")
            encoder_image_bias = tf.Variable(tf.zeros([self.hparams.hidden_units]), name="encoder_image_bias")

            # image features dim scale
            image_batch = tf.reshape(self.video, shape=(-1, self.hparams.video_features_dim))
            embeded_image = tf.nn.xw_plus_b(image_batch, encoder_image_W, encoder_image_bias)
            # embeded_image.shape=(batch_size, enc_max_len, hidden_units)
            embeded_image = tf.reshape(embeded_image, shape=(-1, self.hparams.enc_max_len, self.hparams.hidden_units))

            # lstm1, lstm2
            lstm1 = tf.contrib.rnn.BasicLSTMCell(self.hparams.hidden_units, state_is_tuple=True)
            lstm2 = tf.contrib.rnn.BasicLSTMCell(self.hparams.hidden_units, state_is_tuple=True)
            state1 = lstm1.zero_state(self.hparams.batch_size, dtype=tf.float32)
            state2 = lstm2.zero_state(self.hparams.batch_size, dtype=tf.float32)

            encoder_outputs = []  # item.shape=[batch_size, self.hparams.hidden_units]
            with tf.variable_scope("roll"):
                for i in xrange(self.hparams.enc_max_len):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope("lstm1"):
                        output1, state1 = lstm1(embeded_image[:, i, :], state1)
                    with tf.variable_scope("lstm2"):
                        # output2 batch_size * hidden_units
                        output2, state2 = lstm2(output1, state2)
                        encoder_outputs.append(output2)

        encoder_outputs = map(lambda x: tf.expand_dims(x, axis=2), encoder_outputs)
        # batch_size, time_step, hidden_units
        self.encoder_outputs = tf.transpose(tf.concat(encoder_outputs, axis=2), [0, 2, 1])

        hidden_units = self.hparams.hidden_units
        enc_max_len = self.hparams.enc_max_len

        # 2, encoder_features attention
        with tf.variable_scope("encoder_attention"):
            W_h = tf.get_variable("W_h", shape=[hidden_units, hidden_units], dtype=tf.float32)
            # batch_size x enc_max_len, hidden_units
            flatten_outputs = tf.reshape(self.encoder_outputs, shape=(-1, hidden_units))
            encoder_features = tf.matmul(flatten_outputs, W_h)
            # batch_size, enc_max_len, hidden_units
            self.encoder_features = tf.reshape(encoder_features, shape=(-1, enc_max_len, hidden_units))

        # probs is a list to store the output per decode step, and this is concat for each element
        # loss is a scalar, but it will change to a tensor
        probs = []
        loss = 0.

        # 3, decoder
        with tf.variable_scope("decoder"):
            with tf.device("/cpu:0"):
                embedding_matrix = tf.get_variable("word_embedding", (self.hparams.vocab_size, self.hparams.word_embed_dim))
                # caption_input[batch_size, dec_max_len] <sos> x x x x x <eos>
                embeded_caption = tf.nn.embedding_lookup(embedding_matrix, self.caption_input)

            lstm3 = tf.contrib.rnn.BasicLSTMCell(self.hparams.hidden_units, state_is_tuple=True)
            lstm4 = tf.contrib.rnn.BasicLSTMCell(self.hparams.hidden_units, state_is_tuple=True)

            for i in xrange(self.hparams.dec_max_len):

                if self.hparams.use_attn:       # must use attention
                    attention1 = self._compute_attention(state1, scope_name="attention1", idx=i)
                    attention2 = self._compute_attention(state2, scope_name="attention2", idx=i)
                    if self.infer_mode:
                        if i == 0:
                            # 1,
                            current_input = tf.expand_dims(self.hparams.word2id[self.hparams.sos], -1)
                        else:
                            # probs is (batch_size, vocab_size,) then argmax后 (batch_size,)
                            current_input = tf.argmax(probs[-1], axis=1)
                        current_embedding = tf.nn.embedding_lookup(embedding_matrix, current_input)
                        output3, state1 = lstm3(tf.concat([attention1, current_embedding], -1), state1)
                        output4, state2 = lstm4(tf.concat([attention2, output3], -1), state2)
                    else:
                        output3, state1 = lstm3(tf.concat([attention1, embeded_caption[:, i, :]], -1), state1)
                        output4, state2 = lstm4(tf.concat([attention2, output3], -1), state2)  # batch_size, hidden_units
                else:
                    output3, state1 = lstm3(embeded_caption[:, i, :], state1)
                    output4, state2 = lstm4(output3, state2)  # batch_size, hidden_units

                with tf.variable_scope("decoder_out_projection"):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    output_w = tf.get_variable(name="projection_w", shape=[hidden_units, self.hparams.vocab_size],
                                               initializer=tf.random_normal_initializer())
                    output_b = tf.get_variable(name="projection_b", shape=[self.hparams.vocab_size],
                                               initializer=tf.zeros_initializer())
                    logit_words = tf.nn.xw_plus_b(output4, output_w, output_b)   # batch_size, vocab_size

                probs.append(logit_words)

                if not self.infer_mode:
                    target = self.caption_target[:, i]   # shape=(batch_size,)
                    # target (batch_size, )
                    # logit_words (batch_size, vocab_size)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit_words)
                    # cross_entropy.shape=(batch_size,)
                    cross_entropy = cross_entropy * self.caption_mask[:, i]

                    current_loss = tf.reduce_mean(cross_entropy)
                    loss += current_loss

        # origin: batch_size, vocab_size  new: batch_size, vocab_size, time_step
        probs = tf.concat(map(lambda x: tf.expand_dims(x, axis=2), probs), axis=2)
        # if infer_mode loss=0.
        return loss, probs

    def add_generator(self):
        """this is used for the infer mode"""
        pass

    def reduce_state(self, state):
        with tf.variable_scope("reduce_state", reuse=tf.AUTO_REUSE):
            state_c = state.c
            state_h = state.h
            state = tf.concat((state_c, state_h), axis=1)
            new_dim, ori_dim = state_c.shape[-1], state.shape[-1]
            state_W = tf.get_variable("state_W", shape=(ori_dim, new_dim))
            state_b = tf.get_variable("state_b", shape=(new_dim, ))
            return tf.nn.xw_plus_b(state, state_W, state_b)



















