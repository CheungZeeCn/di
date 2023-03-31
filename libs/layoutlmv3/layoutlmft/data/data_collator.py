import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    # _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy
import numpy as np

from typing import NewType

InputDataClass = NewType("InputDataClass", Any)


def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span


@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                         for position_id in batch["position_ids"]]

        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                         + position_id for position_id in batch["position_ids"]]

        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (
                        sequence_length - len(batch['segment_ids'][i])) + [
                                              batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)

        return batch


@dataclass
class DataCollatorForKeyValueExtractionGp(DataCollatorMixin):
    """
        针对 t1 格式, 不大好顺带处理test的情况.
        适配xfund 的ner任务 也OK.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    num_labels: int = 1
    num_rel_labels: Optional[int] = None
    fixed_text_length: Optional[int] = None
    rel_label_name: Optional[str] = 'rel_labels'
    is_ent_task_only: Optional[bool] = False

    def __call__(self, features):
        # todo: labels 的变化需要看看gp怎么处理
        has_rel_label = True if self.rel_label_name in features[0].keys() else False
        has_label = False
        label_name = "label" if "label" in features[0].keys() else "labels"
        if label_name in features[0].keys():
            has_label = True
        labels = [feature[label_name] for feature in features] if has_label else None
        rel_labels = [feature[self.rel_label_name] for feature in features] if has_rel_label else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        # 保证长度为max_length, 仅仅会pad某些tokenizer内置的字段
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            # return_tensors="pt" if labels is None else None,
            return_tensors=None
        )
        # 这个时候 bbox segment_ids position_ids 不会被pad
        # labels 也不会

        # 处理 images 和对应的 attention mask
        if images is not None:
            batch["images"] = images
            # 这个写法怪得很
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        # 前置一些操作

        # todo: 对比下test的操作 是否可以直接 返回?
        if labels is not None:
            # labels的转化在gp这块非常简单:
            batch_size = len(batch['input_ids'])
            # labels 本来的结构[[[i,j, lable_id], ...], ...]
            # shape (b, num_labels, l, l)
            np_labels = np.zeros([batch_size, self.num_labels, self.fixed_text_length, self.fixed_text_length])

            # 实体label
            for i, i_label_li in enumerate(labels):
                for l_i, l_j, l_id in i_label_li:
                    np_labels[i, l_id, l_i, l_j] = 1
            # 关系label
            if (rel_labels is not None) and (self.is_ent_task_only is not True):
                np_rel_labels = np.zeros(
                    [batch_size, self.num_rel_labels * 2, self.fixed_text_length, self.fixed_text_length])
                # [batch_size, self.num_rel_labels * 2, self.fixed_text_length, self.fixed_text_length], dtype = np.int64)
                for i in range(len(rel_labels)):
                    i_rel_labels = rel_labels[i]
                    i_seq_len = i_rel_labels.shape[-1]
                    np_rel_labels[i, :, :i_seq_len, :i_seq_len] = rel_labels[i]
                # 将实体和关系的label合并
                np_labels = np.concatenate([np_labels, np_rel_labels], axis=-3)
            # 横竖都要删掉这个, 如果有rel label 信息 应在在上一步被 concat到 label里面;
            if self.rel_label_name in batch:
                del batch[self.rel_label_name]
            labels = torch.tensor(np_labels, dtype=torch.long)
            batch["labels"] = labels

        # labels 相关补全, 如果有label
        # sequence_length == 512 ?
        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        # 手工 pad bbox 和  position_ids
        # 这部分在没有label的时候 是不是也要pad 一下才好？
        if padding_side == "right":
            # batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                         for position_id in batch["position_ids"]]
        else:
            # 纯粹的就是不想思考这种情况
            raise NotImplementedError("padding left not supported yet")

        # 补齐image的pad
        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (
                        sequence_length - len(batch['segment_ids'][i])) + [
                                              batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        # pass
        # batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        # debug
        new_batch = {}
        for k, v in batch.items():
            # if k == 'rel_labels':
            #     print(k)
            if isinstance(v[0], list):
                new_batch[k] = torch.tensor(v, dtype=torch.int64)
            else:
                new_batch[k] = v
        batch = new_batch
        # {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        # logging.info(f'record_locate: {batch["doc_ids"]}: {batch["in_doc_token_offset"]}')

        # segment_id 本来就是 bbox的计数 加 最后 一个 image
        # 这块ms的实现版本和hf的版本不一致
        if 'segment_ids' in batch:
            # 三维的attn mask， 同一个bbox内可见
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            # 由于 是GP 的情况 所以没有必要做这个事情了
            # visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            # batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)
            pass

        return batch


@dataclass
class DataCollatorForKeyValueExtractionGpForTestOld(DataCollatorMixin):
    """ 时间原因 写个 test的 手工过去掉label 信息"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    num_labels: int = 1
    num_rel_labels: Optional[int] = None
    fixed_text_length: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = None
        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        # 保证长度为max_length
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None
        )
        # 这个时候 bbox segment_ids position_ids 不会被pad
        # labels 也不会

        # 处理 images
        if images is not None:
            batch["images"] = images
            # 这个写法怪得很
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        # 把下文的pad 搬上来看看 奇怪这里没有 做进一步处理
        ###
        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            # batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                         for position_id in batch["position_ids"]]
        else:
            raise NotImplementedError("padding left not supported yet")

        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (
                        sequence_length - len(batch['segment_ids'][i])) + [
                                              batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        new_batch = {}
        for k, v in batch.items():
            if k == 'rel_labels':
                print(k)
            if isinstance(v[0], list):
                new_batch[k] = torch.tensor(v, dtype=torch.int64)
            else:
                new_batch[k] = v
        batch = new_batch
        # {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        # logging.info(f'record_locate: {batch["doc_ids"]}: {batch["in_doc_token_offset"]}')

        # todo  回头有空再review下这块细节，记不住了;
        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            pass
            # visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            # batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)

        return batch
