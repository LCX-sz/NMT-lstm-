import random
import torch
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm
import numpy as np
from config import cfg
from data import BatchData, DataSet
from model import Seq2Seq
from utils import clip_sentences, tensor2sent, cal_bleu, list2sent
torch.manual_seed(2020)

#for each batch to train
def train(train_data, dev_data, model, optimizer, epoch):
    model.train()
    step = 0.
    epoch_loss = 0.
    batch_loss = 0.
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    for i in tqdm(list(range(len(train_data)))):
        optimizer.zero_grad()
        src_tensor = train_data[indices[i]]['src_tensor'].to(cfg.device)
        src_mask = train_data[indices[i]]['src_mask']
        src_len = src_mask.sum(dim=0).int().squeeze(-1).to(cfg.device)
        tgt_tensor = train_data[indices[i]]['tgt_tensor'].to(cfg.device)
        decoder_outputs = model(src_tensor, src_len, tgt_tensor)
        output = decoder_outputs[1:].view(-1, decoder_outputs.size(-1))
        target = tgt_tensor[1:].view(-1)
        loss = cfg.criterion(output, target)
        loss.backward()
        epoch_loss += loss.item()
        batch_loss += loss.item()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(batch_loss / 20)
            batch_loss = 0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        step += 1
    epoch_loss /= step
    score = evaluate(dev_data, model)
    return epoch_loss, score


# def evaluate(dev_data, model):
#     model.eval()
#     predict_sents = []
#     gold_sents = []
#     random_batch_ids = [random.randint(0, len(dev_data) - 1) for _ in range(cfg.random_batch_num)]
#     step=0
#     for item in tqdm(dev_data):
#         src_tensor = item['src_tensor'].to(cfg.device)
#         src_mask = item['src_mask']
#         src_len = src_mask.sum(dim=0).int().squeeze(-1).to(cfg.device)
#         tgt_tensor = item['tgt_tensor'].to(cfg.device)
#         decoder_path_list = model.beam_decode(src_tensor,src_len)
#         if step in random_batch_ids:
#           predict_sents.extend(list2sent(decoder_path_list,cfg.tgt_id2word))
#           gold_sents.extend(tensor2sent(tgt_tensor[1:].transpose(1, 0), cfg.tgt_id2word))
#         step+=1
#     gold_sents = clip_sentences(gold_sents)
#     predict_sents = clip_sentences(predict_sents)
#     random_sentence_ids = [random.randint(0, len(gold_sents) - 1) for _ in range(cfg.random_sentence_num)]
#     for i in random_sentence_ids:
#         print('-----------')
#         print(gold_sents[i])
#         print(predict_sents[i])
#     score = cal_bleu(gold_sents, predict_sents)
#     return score

def evaluate(dev_data, model):
    model.eval()
    predict_sents = []
    gold_sents = []
    random_batch_ids = [random.randint(0, len(dev_data) - 1) for _ in range(cfg.random_batch_num)]
    step = 0.
    for item in tqdm(dev_data):
        src_tensor = item['src_tensor'].to(cfg.device)
        src_mask = item['src_mask']
        src_len = src_mask.sum(dim=0).int().squeeze(-1).to(cfg.device)
        padded_tgt_tensor = torch.zeros(cfg.max_len + 1, cfg.batch_size, dtype=torch.long).to(cfg.device)
        tgt_tensor = item['tgt_tensor'].to(cfg.device)
        padded_tgt_tensor[:tgt_tensor.size(0)] = tgt_tensor
        decoder_outputs = model.greedy_decode(src_tensor, src_len)
        probs, predict_ids = decoder_outputs.topk(1, dim=-1)
        if step in random_batch_ids:
            predict_sents.extend(tensor2sent(predict_ids.squeeze(-1).transpose(1, 0), cfg.tgt_id2word))
            gold_sents.extend(tensor2sent(tgt_tensor[1:].transpose(1, 0), cfg.tgt_id2word))
        step += 1
    gold_sents = clip_sentences(gold_sents)
    predict_sents = clip_sentences(predict_sents)
    random_sentence_ids = [random.randint(0, len(gold_sents) - 1) for _ in range(cfg.random_sentence_num)]
    for i in random_sentence_ids:
        print('-----------')
        print(gold_sents[i])
        print(predict_sents[i])
    score = cal_bleu(gold_sents, predict_sents)
    return score


def trainIters(train_data_set, dev_data_set, model):
    #model.load_state_dict(torch.load('model3.pt',,map_location='cpu')['model_dict'])
    optimizer = optim.Adam(model.parameters(),lr=cfg.lr)
    # optimizer.load_state_dict(torch.load('model3.pt')['optimizer_dict'])
    patience = cfg.patience
    max_eval_bleu = 0.
    writer = SummaryWriter(log_dir='result')
    for epoch in range(1, cfg.epochs + 1):
        train_batch_data = BatchData(train_data_set).batches_data
        dev_batch_data = BatchData(dev_data_set).batches_data
        train_loss, score = train(train_batch_data, dev_batch_data, model, optimizer, epoch)
        print('\nepoch\t{}train_loss\t{}\tBLEU\t{}'.format(epoch, train_loss, score))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('bleu_score', score, epoch)
        if score > max_eval_bleu:
            max_eval_bleu = score
            patience = cfg.patience
            state_dict = {'epoch': epoch, 'model_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict()}
            torch.save(state_dict, 'D:\\NMT\\modelX.pt')
            print('model saved in {} epoch'.format(epoch))
        if patience < 0:
            writer.close()
            break
    writer.close()
    # dev_batch_data = BatchData(dev_data_set).batches_data
    # score=evaluate(dev_batch_data,model)
    # print(score)


def predict(sentence, model):
  import nltk
  lemmatizer=nltk.WordNetLemmatizer()
  words=[]
  for word in nltk.word_tokenize(sentence.lower()):
            words.append( lemmatizer.lemmatize(word))
  ids = [cfg.sos_token]
  for word in words:
      ids.append(cfg.src_word2id.get(word, cfg.unk_token))
  ids.append(cfg.eos_token)
  src_tensor = torch.tensor([ids]).view(-1, 1).to(cfg.device)
  src_len=torch.tensor([src_tensor.size(0)]).to(cfg.device)
  print(src_len)
  print(src_tensor)
  decoder_outputs = model.greedy_decode(src_tensor, src_len)
  output = decoder_outputs[1:].view(-1, decoder_outputs.size(-1))
  probs, predict_ids = output.topk(1)
  tgt_sent = []
  for id in predict_ids:
      tgt_sent.append(cfg.tgt_id2word[id.item()])
  tgt_sent = ''.join(clip_sentences([tgt_sent]))
  print(tgt_sent)
  return tgt_sent


def train_module():
    #train_data_set = DataSet('train.txt')
    train_data_set = DataSet('smalltrain.txt')
    #dev_data_set = DataSet('val.txt')
    dev_data_set = DataSet('smallval.txt')
    model = Seq2Seq()
    model.to(cfg.device)
    trainIters(train_data_set, dev_data_set, model)


def apply_module(sent):
    model = Seq2Seq().to(cfg.device)
    #model.load_state_dict(torch.load('model3.pt')['model_dict'])
    model.load_state_dict(torch.load('model3.pt')['model_dict'])
    predict(sent, model)


if __name__ == '__main__':
    train_module()
    sentence = 'slowly and not without struggle , america begin to listen .'
    apply_module(sentence)
