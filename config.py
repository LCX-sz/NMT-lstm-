import torch.nn as nn
import os
import torch
from utils import load_json

class Config(object):
    def __init__(self):
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = 'cpu'
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pad_token = 0
        self.unk_token = 1
        self.sos_token = 2
        self.eos_token = 3
        self.batch_size = 40
        self.hidden_size = 160
        self.print_size = 15
        self.lr = 0.0002
        self.weight_decay = 5e-5
        self.dropout = 0.3
        self.epochs = 1000
        self.attn_dim = 64
        self.patience = 10
        self.max_len = 30
        self.beam_width = 4
        self.queue_size = 1000
        self.random_batch_num = 3
        self.random_sentence_num = 10
        self.src_num_layers = 2
        self.tgt_num_layers = 1
        self.teacher_forcing_ratio = 0.5
        self.src_embed_size = 100
        self.src_embed_path = './data/vec/en_vec.npy'
        self.tgt_embed_size = 100
        self.tgt_embed_path = './data/vec/ch_vec.npy'
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.src_vocab_path = 'word_en_vocab.json'
        self.tgt_vocab_path = 'word_ch_vocab.json'
        self.src_word2id = load_json(self.src_vocab_path)
        self.tgt_word2id = load_json(self.tgt_vocab_path)
        self.src_vocab_size = len(self.src_word2id)
        self.tgt_vocab_size = len(self.tgt_word2id)
        self.src_id2word = {v: k for k, v in self.src_word2id.items()}
        self.tgt_id2word = {v: k for k, v in self.tgt_word2id.items()}


cfg = Config()

