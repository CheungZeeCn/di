"""
    modefied by zhangz
    结合网络上的实现做了些变更，这里面有两个版本的Global Pointer

"""
import sys
import math
# from common.utils import Preprocessor
import torch
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


@dataclass
class GPModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    ent_loss: Optional[torch.FloatTensor] = None
    head_loss: Optional[torch.FloatTensor] = None
    tail_loss: Optional[torch.FloatTensor] = None

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
        # debug
        print(
            f"######################################### {y_pred.shape} {(y_pred > 0).sum()} | {y_true.shape} {(y_true > 0).sum()}")

        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            #if end >= start:
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        assert Z != 0
        if Y == 0:
            precision = 0
        else:
            precision = X / Y
        f1, recall = 2 * X / (Y + Z), X / Z
        return f1, precision, recall


def get_sinusoid_encoding_table(n_position, d_hid, coord_dim=1):
    ''' sinusoid编码
        这个是经典做法 log 和 exp 提升精度
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :return: [seq_len, d_hid]
            拿到的就是transformer 的那个位置编码
            长度为n_position, 每一维的值对应一条三角函数，sin / cos 交替，频率越来越低, 波形越来越平缓
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    d_hid //= coord_dim
    # 维度
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def __init__(self, max_position, embedding_size, coord_dim=1):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size,
                                                          coord_dim=coord_dim)  # [seq_len, hdsz]
        # 复制 * 2
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        self.coord_dim = coord_dim
        # buffer 后 这组参数不会被更新 而且存模型的时候也会被存进去
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, pos=None, seq_dim=-3):
        # 默认最后几个维度为[seq_len, types, hdsz]
        seq_len = qw.shape[seq_dim]
        if self.coord_dim == 1:
            pos = torch.range(0, seq_len - 1).long().reshape(-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
            return qw * (self.cos_position[pos].unsqueeze(-2)) + qw2 * (self.sin_position[pos].unsqueeze(-2))
        elif pos is not None and self.coord_dim == 2:
            # (batch_size, 512)
            x_pos = pos[0]
            y_pos = pos[1]

            # todo 直接做一个矩阵版本 而不是for循环 提升性能
            all_cos_pos = []
            all_sin_pos = []
            for i_batch_position_ids in (x_pos, y_pos):
                all_cos_pos.append(self.cos_position[i_batch_position_ids])
                all_sin_pos.append(self.sin_position[i_batch_position_ids])
            batch_cos_pos = torch.concat(all_cos_pos, dim=-1).unsqueeze(-2)
            batch_sin_pos = torch.concat(all_sin_pos, dim=-1).unsqueeze(-2)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
            return qw * batch_cos_pos + qw2 * batch_sin_pos
        else:
            raise NotImplementedError("RoPE 出错, 检查下")


class GlobalPointerLayer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True,
                 coord_dim=1):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        self.coord_dim = coord_dim
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size, coord_dim=coord_dim)

    def forward(self, inputs, pos=None, mask=None):
        '''
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        '''
        sequence_output = self.dense(inputs)  # [..., heads*head_size*2]
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1),
                                      dim=-2)  # [..., heads, head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw, pos=pos)
            kw = self.position_embedding(kw, pos=pos)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class GlobalPointerRelModel(nn.Module):
    """
        用于NER
        这一版本先做简单的，对实体类型不敏感的, 只区分了头实体、尾实体;
        todo: ent_type_size != 2 的情况，就是将KV的类型区分放到模型中，针对任务来定制; 对比下精度.
    """

    def __init__(self, encoder, config, rel_type_size, ent_type_size=2, inner_dim=64, RoPE=True, RoPE_dim=2,
                 fixed_text_len=None):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.rel_type_size = rel_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.fixed_text_len = fixed_text_len
        self.config = config

        self.RoPE = RoPE
        self.RoPE_dim = RoPE_dim

        # 实体 head, tail
        self.entity_output = GlobalPointerLayer(self.hidden_size, ent_type_size, inner_dim, RoPE=RoPE,
                                                coord_dim=RoPE_dim,
                                                max_len=1024, use_bias=True, tril_mask=True)
        # 关系
        self.head_output = GlobalPointerLayer(self.hidden_size, rel_type_size, inner_dim, RoPE=RoPE, coord_dim=RoPE_dim,
                                              max_len=1024, use_bias=True, tril_mask=False)
        self.tail_output = GlobalPointerLayer(self.hidden_size, rel_type_size, inner_dim, RoPE=RoPE, coord_dim=RoPE_dim,
                                              max_len=1024, use_bias=True, tril_mask=False)

    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size+rel_type_size*2, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size+rel_type_size*2, seq_len, seq_len)
        """
        assert y_true.shape[2] == y_true.shape[3] == y_pred.shape[2] == y_pred.shape[
            3], f"asset: 'y_true.shape[2] {y_true.shape[2]} == y_true.shape[3] {y_true.shape[3]} == y_pred.shape[2] {y_pred.shape[2]} == y_pred.shape[3] {y_pred.shape[3]}' failed"
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                valid_span=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                bbox=None,
                images=None,
                in_doc_token_offset=None,
                doc_ids=None,
                ):
        """
            兼容transformers
            在训练环节上和实体的类似，就是label的处理方法不同
        """

        self.device = input_ids.device
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            valid_span=valid_span,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_values=None,
            # use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        # context_outputs = self.dropout(sequence_output)
        context_outputs = sequence_output

        if isinstance(context_outputs, dict):
            last_hidden_state = context_outputs['last_hidden_state']
        else:
            # last_hidden_state:(batch_size, seq_len, hidden_size)
            last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        if self.fixed_text_len is not None:
            seq_len = self.fixed_text_len
            # 仅仅取文本部分的hidden_state
            last_hidden_state = last_hidden_state[:, 0:self.fixed_text_len, :]
            attention_mask = attention_mask[:, 0:self.fixed_text_len]
        else:
            seq_len = last_hidden_state.size()[1]

        x_pos = bbox[:, :, 0]
        y_pos = bbox[:, :, 1]

        if self.RoPE_dim == 1:
            ent_logits = self.entity_output(last_hidden_state, attention_mask)
            head_logits = self.head_output(last_hidden_state, attention_mask)
            tail_logits = self.tail_output(last_hidden_state, attention_mask)
        elif self.RoPE_dim == 2:
            # 二维
            ent_logits = self.entity_output(last_hidden_state, pos=(x_pos, y_pos), mask=attention_mask)
            head_logits = self.head_output(last_hidden_state, pos=(x_pos, y_pos), mask=attention_mask)
            tail_logits = self.tail_output(last_hidden_state, pos=(x_pos, y_pos), mask=attention_mask)

        # 其实这块完全可以合并啊... 其实就是heads 变多了而已
        logits = torch.cat([ent_logits, head_logits, tail_logits], dim=-3)

        # 先不算loss 让外层算
        if labels is not None:
            assert logits.shape == labels.shape, f"{logits.shape} == {labels.shape}"

        loss, ent_loss, head_loss, tail_loss = None, None, None, None
        if labels is not None:
            # debug
            if len(set([labels.shape[2], labels.shape[3], logits.shape[2] , logits.shape[3]])) != 1:
                print("ht")
            loss = self.loss_fun(labels, logits)

            ent_loss = self.loss_fun(labels[:, :self.ent_type_size, :, :], logits[:, :self.ent_type_size, :, :])

            head_loss = self.loss_fun(labels[:, self.ent_type_size:self.ent_type_size + self.rel_type_size, :, :],
                                      logits[:, self.ent_type_size:self.ent_type_size + self.rel_type_size, :, :])
            tail_loss = self.loss_fun(labels[:, -self.rel_type_size:, :, :], logits[:, -self.rel_type_size:, :, :])


        if not return_dict:
            output = [logits, last_hidden_state]
            return ([loss,] + output) if loss is not None else output
        else:
            return GPModelOutput(
                loss=loss,
                logits=logits,
                last_hidden_state=last_hidden_state,
                # ent_loss=ent_loss,
                # head_loss=head_loss,
                # tail_loss=tail_loss
            )

class GlobalPointerNerModel(nn.Module):
    """
        用于NER
    """

    def __init__(self, encoder, config, ent_type_size, inner_dim=64, RoPE=True, RoPE_dim=2, fixed_text_len=None):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.fixed_text_len = fixed_text_len
        self.config = config

        self.RoPE = RoPE
        self.RoPE_dim = RoPE_dim
        self.global_pointer = GlobalPointerLayer(self.hidden_size, ent_type_size, inner_dim, RoPE=RoPE,
                                                 coord_dim=RoPE_dim,
                                                 max_len=1024, use_bias=True, tril_mask=True)

    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        assert y_true.shape[2] == y_true.shape[3] == y_pred.shape[2] == y_pred.shape[
            3], "asset: 'y_true.shape[2] == y_true.shape[3] == y_pred.shape[2] == y_pred.shape[3]' failed"
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                valid_span=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                bbox=None,
                images=None,
                in_doc_token_offset=None,
                doc_ids=None,
                ):
        """
            兼容transformers
        """

        self.device = input_ids.device
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            valid_span=valid_span,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_values=None,
            # use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        # context_outputs = self.dropout(sequence_output)
        context_outputs = sequence_output

        if isinstance(context_outputs, dict):
            last_hidden_state = context_outputs['last_hidden_state']
        else:
            # last_hidden_state:(batch_size, seq_len, hidden_size)
            last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        if self.fixed_text_len is not None:
            seq_len = self.fixed_text_len
            # 仅仅取文本部分的hidden_state
            last_hidden_state = last_hidden_state[:, 0:self.fixed_text_len, :]
            attention_mask = attention_mask[:, 0:self.fixed_text_len]
        else:
            seq_len = last_hidden_state.size()[1]

        x_pos = bbox[:, :, 0]
        y_pos = bbox[:, :, 1]

        if self.RoPE_dim == 1:
            logits = self.global_pointer(last_hidden_state, attention_mask)
        else:
            logits = self.global_pointer(last_hidden_state, pos=(x_pos, y_pos), mask=attention_mask)

        # 先不算loss 让外层算
        assert logits.shape == labels.shape, f"{logits.shape} == {labels.shape}"

        loss = None
        if labels is not None:
            loss = self.loss_fun(labels, logits)

        if not return_dict:
            output = (logits, last_hidden_state)
            return ((loss,) + output) if loss is not None else output
        else:
            return GPModelOutput(
                loss=loss,
                logits=logits,
                last_hidden_state=last_hidden_state
            )


class GlobalPointerModel(nn.Module):
    """
        修改了网络上的一个版本，
        这个实现有点死板, 可以用model来包一个layer 这样更灵活一些
        仅对接layoutlmv3
    """
    def __init__(self, encoder, config, ent_type_size, inner_dim=64, RoPE=True, RoPE_dim=2, fixed_text_len=None):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.fixed_text_len = fixed_text_len
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        self.RoPE = RoPE
        self.RoPE_dim = RoPE_dim

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        """
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def single_sinusoidal_position_embedding(self, seq_len, dim, coord_dim=1):
        """
            计算一根 emb, coord_dim 默认是1 返回的向量shape: (seq_len, dim/2, 2)
            最后一维度，一份是sin的值，一份是cos的值
            coord_dim 为2 时候 shape: (seq_len, dim/4, 2)

        """
        div = coord_dim * 2

        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, dim // div, dtype=torch.float)
        # 分片大小
        indices = torch.pow(10000, -1 * div * indices / dim)

        # 数值计算 0, 1, 2, 3, 4, ...  seq_len
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.to(self.device)
        embeddings = embeddings.reshape(seq_len, -1)
        return embeddings

    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        assert y_true.shape[2] == y_true.shape[3] == y_pred.shape[2] == y_pred.shape[
            3], "asset: 'y_true.shape[2] == y_true.shape[3] == y_pred.shape[2] == y_pred.shape[3]' failed"
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                valid_span=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                bbox=None,
                images=None):

        self.device = input_ids.device
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            valid_span=valid_span,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_values=None,
            # use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        # context_outputs = self.dropout(sequence_output)
        context_outputs = sequence_output

        if isinstance(context_outputs, dict):
            last_hidden_state = context_outputs['last_hidden_state']
        else:
            # last_hidden_state:(batch_size, seq_len, hidden_size)
            last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        if self.fixed_text_len is not None:
            seq_len = self.fixed_text_len
            # 仅仅取文本部分的hidden_state
            last_hidden_state = last_hidden_state[:, 0:self.fixed_text_len, :]
            attention_mask = attention_mask[:, 0:self.fixed_text_len]
        else:
            seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            if self.RoPE_dim == 1:
                # pos_emb:(batch_size, seq_len, inner_dim)
                pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
                # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
                cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
                qw2 = qw2.reshape(qw.shape)
                qw = qw * cos_pos + qw2 * sin_pos
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
                kw2 = kw2.reshape(kw.shape)
                kw = kw * cos_pos + kw2 * sin_pos
            elif self.RoPE_dim == 2:
                # 原始的曲线坐标
                # pos_emb.shape: (1024, inner_dim/(2*coord_dim))
                pos_emb = self.single_sinusoidal_position_embedding(1024, self.inner_dim, coord_dim=2)

                # (1024, inner_dim/(2*coord_dim))
                cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
                sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

                # (batch_size, 512)
                x_pos = bbox[:, :, 0]
                y_pos = bbox[:, :, 1]

                # todo 直接做一个矩阵版本 而不是for循环 提升性能
                all_cos_pos = []
                all_sin_pos = []
                for i_batch_position_ids in (x_pos, y_pos):
                    all_cos_pos.append(cos_pos[i_batch_position_ids])
                    all_sin_pos.append(sin_pos[i_batch_position_ids])
                batch_cos_pos = torch.concat(all_cos_pos, dim=-1).unsqueeze(-2)
                batch_sin_pos = torch.concat(all_sin_pos, dim=-1).unsqueeze(-2)

                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
                qw2 = qw2.reshape(qw.shape)
                qw = qw * batch_cos_pos + qw2 * batch_sin_pos
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
                kw2 = kw2.reshape(kw.shape)
                kw = kw * batch_cos_pos + kw2 * batch_sin_pos
                assert kw.shape == qw.shape
                assert batch_sin_pos.shape == batch_cos_pos.shape

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask cut
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角 如果用在关系抽取 就不能排除
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        # scale
        logits = logits / self.inner_dim ** 0.5

        # 先不算loss 让外层算
        assert logits.shape == labels.shape, f"{logits.shape} == {labels.shape}"

        loss = None
        if labels is not None:
            loss = self.loss_fun(labels, logits)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        else:
            return GPModelOutput(
                loss=loss,
                logits=logits,
                last_hidden_state=last_hidden_state
            )


