# -*- coding: utf-8 -*-
# @Author: lidongdong
# @time  : 18-12-1 下午3:16
# @file  : hparams.py


class HParams(object):
    def __init__(self):

        # model
        self.hidden_units = 256
        self.word_embed_dim = 128
        self.encoder_layers = 2
        self.decoder_layers = 2
        self.encoder_type = "bi"    # {bi, uni}
        self.unit_type = "lstm"     # {lstm, gru}

        # model train
        self.optimizer = "sgd"
        # lr 设置为0.005, 然后每过1000个batch, 将lr变为原来的0.95
        # learning_rate = 0.0001
        self.learning_rate = 0.005
        self.learning_decay = 0.95
        self.decay_interval = 3000
        self.initial_weight = 0.1
        self.train_steps = 100010
        self.steps = 100
        self.batch_size = 30
        self.save_interval = 5000

        # model eval
        self.beam_width = 5

        # misc
        self.enc_max_len = 30
        self.dec_max_len = 20
        self.sos = "<s>"
        self.eos = "</s>"
        self.unk = "<unk>"
        self.video_features_dim = 4096
        self.max_grad_norm = 1.
        self.summary_path = "/tmp/s2vt"
        self.eval_interval = 1000            # eval per after 200 train step
        self.use_attn = True

        # data
        self.train_data = "data_preprocess/vgg/train_data"
        self.test_data = "data_preprocess/vgg/test_data"
        self.vocab_path = "data_preprocess/vocab.txt"
        self.csv_file="video_processed_2.csv"
        # >>> sam.shape
        # (100, 30, 4096)
        # >>> np.mean(sam)
        # -0.75592643
        # >>> np.var(sam)
        # 4.5537667
        self.mean = -0.7559
        self.var = 4.5537

        # vocab
        self.word2id, self.id2word = self.get_vocab()
        self.vocab_size = len(self.word2id)

    def get_vocab(self, vocab_path=None):
        if vocab_path is None:
            vocab_path = self.vocab_path

        vocab = {}

        for ix, word in enumerate([self.unk, self.sos, self.eos]):
            vocab[word] = ix

        line_num = 0
        for line in open(vocab_path):
            line = line.strip()
            if line == "": break
            pieces = line.split()
            assert len(pieces) == 2, \
                "wrong format in {line} line of file {file}".format(line=line_num + 1, file=vocab_path)
            vocab[pieces[0]] = line_num + 3
            line_num += 1

        id2word = dict(zip(vocab.values(), vocab.keys()))
        return vocab, id2word
