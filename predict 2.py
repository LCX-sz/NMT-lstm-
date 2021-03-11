from PySide2.QtWidgets import QApplication, QMessageBox, QMainWindow,QMessageBox,QRadioButton
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from io import open
import pickle
import jieba
import jieba.analyse
import random
import torch
import os
import PySide2
#from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm
import numpy as np
from config import cfg
from data import BatchData, DataSet
from model import Seq2Seq
from utils import clip_sentences, tensor2sent, cal_bleu, list2sent
torch.manual_seed(2020)

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print(plugin_path)
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


torch.cuda.empty_cache()
with torch.no_grad():
 model = Seq2Seq().to(cfg.device)
 model.load_state_dict(torch.load('D:\\NMT\\model.pt',map_location='cpu')['model_dict'])
 #model.load_state_dict(torch.load('D:\\NMT\\model.pt',map_location='cuda:0')['model_dict'])

class Stats:
    def __init__(self):
        self.ui = QUiLoader().load('F:\\Msc_project.ui')
        self.ui.PushButton.clicked.connect(self.handleCalc)
        self.ui.pushButton_2.clicked.connect(self.handleCalc2)
        self.window = QMainWindow()
        self.window.resize(500, 400)
        self.window.move(450, 200)
        self.ui.radioButton.clicked.connect(self.handleCalc3)
        self.ui.radioButton_2.clicked.connect(self.handleCalc4)
        self.pattern=1

    def handleCalc(self):
        sent = self.ui.plainTextEdit.toPlainText()
        length=len(sent)
        print(length)
        if length==0:
            QMessageBox.about(self.window,'warnning:empty input','please write something!')
        elif(length!=0 and self.pattern==1):
        # model = Seq2Seq().to(cfg.device)
        # model.load_state_dict(torch.load('D:\\NMT\\model.pt',map_location='cpu')['model_dict'])
            result=predict(sent, model)
            self.ui.plainTextEdit_2.appendPlainText(result)
        else:
            QMessageBox.about(self.window, 'warnning:change the src and target', 'will coming')


    def handleCalc2(self):
        self.ui.plainTextEdit_2.clear()

    def handleCalc3(self):
        self.pattern=1
    def handleCalc4(self):
        self.pattern=2
app = QApplication([])
app.setWindowIcon(QIcon('F:\\xjtlu.png'))
stats = Stats()
stats.ui.show()
app.exec_()