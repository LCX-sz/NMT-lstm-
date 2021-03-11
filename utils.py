import json

from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings('ignore')


def load_json(file):
    fr = open(file, 'r', encoding='utf-8')
    return json.load(fr)


def clip_sentences(sentences):
    clipped_sentences = []
    clipped_sent = []
    for sent in sentences:
        for word in sent:
            if word == '<eos>':
                break
            else:
                clipped_sent.append(word)
        clipped_sent = clipped_sent if len(clipped_sent) > 1 else ['<empty>']
        clipped_sentences.append(' '.join(clipped_sent))
        clipped_sent = []
    return clipped_sentences


def tensor2sent(tensor, id2word):
    sentences = []
    for row in tensor:
        sent = []
        for column in row:
            sent.append(id2word[column.item()])
        sentences.append(sent)
    return sentences


def list2sent(path_list, id2word):
    sentences = []
    for path in path_list:
        sent = []
        for word_id in path:
            sent.append(id2word[word_id])
        sentences.append(sent)
    return sentences


def cal_bleu(gold_sentences, predict_sentences):
    score = 0.
    for gold_sent, predict_sent in zip(gold_sentences, predict_sentences):
        reference = gold_sent.split(' ')
        hypothesis = predict_sent.split(' ')
        score += sentence_bleu([reference], hypothesis)
        s = sentence_bleu([reference], hypothesis)
        if s >= 0.3:
            print('----------------------------------------------')
            print(gold_sent, predict_sent)
    score /= len(gold_sentences)
    return round(score, 4)
