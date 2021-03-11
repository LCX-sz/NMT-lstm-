import random
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class Encoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.input_dim = cfg.src_vocab_size
        self.emb_dim = cfg.src_embed_size
        self.enc_hid_dim = cfg.hidden_size
        self.dec_hid_dim = cfg.hidden_size
        self.dropout = cfg.dropout

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)

        self.rnn = nn.GRU(self.emb_dim, self.enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.enc_hid_dim = cfg.hidden_size
        self.dec_hid_dim = cfg.hidden_size

        self.attn_in = (self.enc_hid_dim * 2) +self.dec_hid_dim

        self.attn = nn.Linear(self.attn_in, cfg.attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,attention,cfg):
        super().__init__()

        self.emb_dim = cfg.tgt_embed_size
        self.enc_hid_dim = cfg.hidden_size
        self.dec_hid_dim = cfg.hidden_size
        self.output_dim = cfg.tgt_vocab_size
        self.dropout = cfg.dropout
        self.attention = attention
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)

        self.rnn = nn.GRU((self.enc_hid_dim * 2) + self.emb_dim, self.dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + self.emb_dim, self.output_dim)

        self.dropout = nn.Dropout(self.dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))

        return output, decoder_hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,cfg):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device=cfg.device
        self.cfg=cfg

    def forward(self,
                src: Tensor,
                trg=None,
                teacher_forcing_ratio: float = 1.0) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else self.cfg.max_len
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        output = trg[0,:] if trg is not None else torch.tensor([self.cfg.sos_token for i in range(batch_size)], dtype=torch.long).to(self.cfg.device)

        for t in range(0, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
