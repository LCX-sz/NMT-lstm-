import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from config import cfg


class DataSet:
    def __init__(self, corpus_file, clip_size=None):
        self.corpus_file = corpus_file
        self.clip_size = clip_size
        self.examples = self.load_dataset()

    def load_dataset(self):
        dataset = []
        fr = open(self.corpus_file, 'r', encoding='utf-8')
        for line in fr.readlines():
            segments = line.strip().split('\t')
            if not len(segments) == 2:
                continue
            if len(segments[0].split(' ')) < cfg.max_len and len(segments[1].split(' ')) < cfg.max_len:
                src_sent = self.en_sentence2ids(segments[0], cfg.src_word2id)
                tgt_sent = self.ch_sentence2ids(segments[1], cfg.tgt_word2id)
                dataset.append([src_sent, tgt_sent])
        if self.clip_size:
            dataset = dataset[:self.clip_size]
        return dataset

    def en_sentence2ids(self, sentence, vocab):
        original_sent = ['<sos>'] + sentence.split(' ') + ['<eos>']
        sent_ids = [cfg.sos_token]
        for word in sentence.split(' '):
            sent_ids.append(vocab.get(word, cfg.unk_token))
        sent_ids.append(cfg.eos_token)
        sent_ids = torch.tensor(sent_ids, dtype=torch.long)
        return (sent_ids, original_sent)

    def ch_sentence2ids(self, sentence, vocab):
        original_sent = ['<sos>'] + sentence.split(' ') + ['<eos>']
        sent_ids = [cfg.sos_token]
        for character in sentence.split(' '):
            if character != '':
                sent_ids.append(vocab.get(character, cfg.unk_token))
        sent_ids.append(cfg.eos_token)
        sent_ids = torch.tensor(sent_ids, dtype=torch.long)
        return (sent_ids, original_sent)


class BatchData:
    def __init__(self, dataset: DataSet):
        self.batches_data = self.get_batch(dataset)

    def get_batch(self, dataset: DataSet):
        batches_data = []
        indices = list(range(len(dataset.examples)))
        np.random.shuffle(indices)
        batch_data = self.init_batch_items()
        for i in range(len(dataset.examples)):
            batch_data['src_tensor'].append(dataset.examples[indices[i]][0][0])
            batch_data['src_content'].append(dataset.examples[indices[i]][0][1])
            batch_data['tgt_tensor'].append(dataset.examples[indices[i]][1][0])
            batch_data['tgt_content'].append(dataset.examples[indices[i]][1][1])
            if (i + 1) % cfg.batch_size == 0:
                batch_data['src_tensor'] = pad_sequence(batch_data['src_tensor'])
                batch_data['src_content'] = batch_data['src_content']
                batch_data['src_mask'] = (batch_data['src_tensor'] != 0).float().unsqueeze(-1)
                batch_data['tgt_tensor'] = pad_sequence(batch_data['tgt_tensor'])
                batch_data['tgt_content'] = batch_data['tgt_content']
                batch_data['tgt_mask'] = (batch_data['tgt_tensor'] != 0).float().unsqueeze(-1)
                batches_data.append(batch_data)
                batch_data = self.init_batch_items()
        return batches_data

    def init_batch_items(self):
        items = ['src_tensor', 'src_content', 'tgt_tensor', 'tgt_content', 'src_mask', 'tgt_mask']
        batch_data = {}
        for item in items:
            batch_data[item] = []
        return batch_data
