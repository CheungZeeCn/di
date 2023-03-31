#!/usr/bin/env python
# coding=utf-8
"""

GP关系抽取
参考参数:
--data_dir /home/ana/data2/datasets/icdar2023.challenge/icdar2023/task1/task1_test
--do_predict
--trained_model_path /home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230316/checkpoint-1000
--config_name /home/ana/data2/models/layoutlmv3-base-chinese
--output_dir t1.rel.test_output.demo
--segment_level_layout 1 --visual_embed 1 --input_size 224
--per_device_train_batch_size 1

"""
import copy
import tqdm
import collections
from collections import defaultdict
import json
import networkx as nx
from collections import Counter

from transformers.trainer_pt_utils import (
    find_batch_size,
)

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# from datasets import ClassLabel, load_dataset, load_metric
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version

import kp_setup
from libs.layoutlmv3.layoutlmft.data import DataCollatorForKeyValueExtractionGp
from libs.datasets.t1_dataset_gp import t1_test_dataset_gp, label2ids_span_gp, rel_types_sp
from libs.layoutlmv3.layoutlmft.models import LayoutLMv3Model
from libs.GlobalPointer import MetricsCalculator, GlobalPointerRelModel

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# todo: 目前的实现版本其实也不能太高，会和官方的layoutlmv3 有冲突，后续看看如何解决
check_min_version("4.5.0")
# root logger
logger = logging.getLogger()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    trained_model_path: str = field(
        default="/home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230316/checkpoint-1000",
        metadata={"help": "训练好的模型"}
    )
    config_name: Optional[str] = field(
        default="/home/ana/data2/models/layoutlmv3-base-chinese/",
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="/home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230316/checkpoint-1000/",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # cache_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    # )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )


@dataclass
class DataTrainingArguments:
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


def get_label_list():
    label_list = [[key, val] for key, val in label2ids_span_gp.items()]
    label_list = sorted(label_list, key=lambda x: x[1], reverse=False)
    label_list = [label for label, id in label_list]
    return label_list

def get_rel_list():
    label_list = [[key, val] for key, val in rel_types_sp.items()]
    label_list = sorted(label_list, key=lambda x: x[1], reverse=False)
    label_list = [label for label, id in label_list]
    return label_list


def dump_rel_result(file_out, rel_result):
    """任务提交格式
   {
    "0.jpg": {
        "KV-Pairs": [
        {
        "Keys": [
        {
        "Content": "prediction_keys"
        }
        ],
        "Values": [
        {
        "Content": "prediction_values"
        }
        ]
        }
        ]
        },
    """
    result_json = {}
    for doc_id, doc_result in rel_result.items():
        relation_infos = doc_result['relations'][0]
        doc_result = []
        for relation_info in relation_infos:
            keys = "\t".join(relation_info[1][0])
            value = relation_info[1][1]
            item = {"Keys": [{"Content": keys}], "Values":[{"Content": value}]}
            doc_result.append(item)
        result_json[doc_id] = {"KV-Pairs":doc_result}

    with open(file_out, "w") as f:
        logging.info("dumping result for {} doc into {}".format(len(result_json), file_out))
        json.dump(result_json, f, ensure_ascii=False, indent=2)


def collect_entity_result(result_collections, dataset, tokenizer, ):
    """
        收集 实体结果
        dataset:
        self.total_inputs = {
            "total_input_ids": total_input_ids,
            "total_bboxs": total_bboxs,
            "total_bbox_label_ids": total_bbox_label_ids,
            "total_span_label_ids": total_span_label_ids,
            "total_token_len": total_token_len,
            "total_span_rel_labels": total_span_rel_labels
        }
            self.doc_id_to_index[cur_doc_id] = i
    """
    entity_label_list = get_label_list()
    entity_results = {}
    for doc_id, doc_results in result_collections.items():
        doc_index = dataset.doc_id_to_index[doc_id]
        # print("doc_index", doc_index)
        # offset 排好序
        doc_results = sorted(doc_results, key=lambda x: x[0])
        doc_entities = defaultdict(list)
        for seg_offset, seg_result in doc_results:
            for ent_type_id, ent_type in enumerate(entity_label_list):
                for b, e in seg_result[ent_type_id]:
                    # 实体
                    if b > e:
                        continue
                    ent_input_ids = dataset.total_inputs["total_input_ids"][doc_index][b:e + 1]
                    ent_text = tokenizer.decode(ent_input_ids, skip_special_tokens=True)
                    ent_bbox = dataset.total_inputs["total_bboxs"][doc_index][b]
                    doc_entities[ent_type].append((ent_text, ent_bbox, b, e))
        entity_results[doc_id] = doc_entities
    return entity_results


def collect_rel_result(result_collections, dataset, tokenizer):
    """
        收集 关系 结果 , 这块涉及任务逻辑比较多
            1. 先收集实体结果;
            2. 处理K_GRP 和 V_GRP 数据, 并生成对应的  GRP 并给予对应的GRP_ID, 关系肯定就在 GRP 中处理; networkx 用起来
            3. 处理 K_V数据

        dataset:
        self.total_inputs = {
            "total_input_ids": total_input_ids,
            "total_bboxs": total_bboxs,
            "total_bbox_label_ids": total_bbox_label_ids,
            "total_span_label_ids": total_span_label_ids,
            "total_token_len": total_token_len,
            "total_span_rel_labels": total_span_rel_labels
        }
            self.doc_id_to_index[cur_doc_id] = i
    """
    entity_label_list = get_label_list()
    rel_label_list = get_rel_list()

    all_result = {}
    entity_results = {}
    relation_results = {}

    for doc_id, doc_results in result_collections.items():
        doc_ent_rel_result = {"entity": [], "relations": []}
        doc_index = dataset.doc_id_to_index[doc_id]
        # print("doc_index", doc_index)
        # offset 排好序
        doc_results = sorted(doc_results, key=lambda x: x[0])
        doc_token_len = dataset.total_inputs['total_token_len'][doc_index]
        doc_input_ids = dataset.total_inputs["total_input_ids"][doc_index]

        # b，e 为 key
        doc_entities = defaultdict(dict)

        doc_relations = defaultdict(set)

        # 坐标为 (b,e)
        doc_key_span_dict = {}
        doc_value_span_dict = {}
        key_groups = {}
        value_groups = {}

        # for 每一个片段 先把实体信息处理好
        for seg_offset, seg_result in doc_results:
            # 看实体
            for ent_type_id, ent_type in enumerate(entity_label_list):
                for b, e in seg_result[ent_type_id]:
                    # 实体
                    if b > e:
                        continue
                    ent_input_ids = dataset.total_inputs["total_input_ids"][doc_index][b:e + 1]
                    ent_text = tokenizer.decode(ent_input_ids, skip_special_tokens=True)
                    ent_bbox = dataset.total_inputs["total_bboxs"][doc_index][b]
                    # 注意 这种方案 pred 的时候 天然的会有overlap 所以做实体的时候可以偏重召回 先都保留 最后输出的时候再策略合并
                    doc_entities[ent_type][(b, e)] = (ent_text, ent_bbox, b, e)

        # 关系_type -> set([(index1, index2), ])
        for seg_offset, seg_result in doc_results:
            # 先收集好关系元组
            for rel_type_id, rel_type in enumerate(rel_label_list):
                # begin->begin
                rel_type_id_begin = len(entity_label_list) + rel_type_id
                # end->end
                rel_type_id_end = rel_type_id_begin + len(rel_label_list)

                for b, e in seg_result[rel_type_id_begin]:
                    doc_relations[rel_type + "_b"].add((b, e))

                for b, e in seg_result[rel_type_id_end]:
                    doc_relations[rel_type + "_e"].add((b, e))

        rel_infos, all_raw_relations, k_grps, v_grps = collect_doc_rel_results_2(doc_entities, doc_relations, tokenizer,
                                                                                 doc_input_ids)

        relation_results[doc_id] = [rel_infos, all_raw_relations, k_grps, v_grps]
        entity_results[doc_id] = doc_entities
        all_result[doc_id] = {'entity': doc_entities, 'relations': relation_results[doc_id]}
    return all_result


def collect_doc_rel_results_2(doc_entities, doc_relations, tokenizer, doc_input_ids):
    """
        特定任务配置特定的关系组装
        return rel_infos, all_raw_relations, k_grps, v_grps
        # 发现有些问题, 直接不区分实体类型了,  感觉如果连头尾实体都不区分?
    """
    # 开始组装关系了
    # K_GRP: k k 循环 然后判断, 有的话就加入关系元组中
    # K-> 1, V -> 2
    # K_GRP
    all_raw_relations = set()
    k_grps = []
    be_2_k_grp = {}
    k_grp_di_g = nx.DiGraph()


    all_entities = copy.deepcopy(doc_entities['KEY'])
    all_entities.update(doc_entities['VALUE'])
    all_entities.update(doc_entities['BACKGROUND'])

    for i_b, i_e in all_entities:
        # 避免有孤立的点
        k_grp_di_g.add_node((i_b, i_e))
        for j_b, j_e in all_entities:
            # 这两个key span 是否有K_GRP 关系?
            if (i_b, j_b) in doc_relations['K_GRP_b'] and (i_e, j_e) in doc_relations["K_GRP_e"]:
                all_raw_relations.add((i_b, i_e, 'K_GRP', j_b, j_e))
                # 节点
                k_grp_di_g.add_edge((i_b, i_e), (j_b, j_e))
    # 组织 k_grp
    k_grp_g = k_grp_di_g.to_undirected()
    for node_sets in nx.connected_components(k_grp_g):
        # 直接按照offset 排序 香得很 先不用 边来排了
        nodes = sorted(list(node_sets), key=lambda x: x[0])
        k_grp_id = len(k_grps)
        k_grps.append(nodes)
        for node in nodes:
            be_2_k_grp[node] = k_grp_id

    # V_GRP 类似
    v_grps = []
    be_2_v_grp = {}
    v_grp_di_g = nx.DiGraph()
    # 2 => V
    for i_b, i_e in all_entities:
        # 避免有孤立的点
        v_grp_di_g.add_node((i_b, i_e))
        for j_b, j_e in all_entities:
            # 这两个key span 是否有K_GRP 关系?
            if (i_b, j_b) in doc_relations['V_GRP_b'] and (i_e, j_e) in doc_relations["V_GRP_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'V_GRP', j_b, j_e, i_str, j_str))
                # 节点
                v_grp_di_g.add_edge((i_b, i_e), (j_b, j_e))

    # 组织 v_grp
    v_grp_g = v_grp_di_g.to_undirected()
    for node_sets in nx.connected_components(v_grp_g):
        # 直接按照offset 排序 香得很 先不用 边来排了
        nodes = sorted(list(node_sets), key=lambda x: x[0])
        v_grp_id = len(v_grps)
        v_grps.append(nodes)
        for node in nodes:
            be_2_v_grp[node] = v_grp_id

    # 难度提升 K_K:  K1\tK2 字样
    kk_grp_rel = set()
    for i_b, i_e in all_entities:
        for j_b, j_e in all_entities:
            if (i_b, j_b) in doc_relations['K_K_b'] and (i_e, j_e) in doc_relations["K_K_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'K_K', j_b, j_e, i_str, j_str))
                # 节点
                i_grp = be_2_k_grp[(i_b, i_e)]
                j_grp = be_2_k_grp[(j_b, j_e)]
                kk_grp_rel.add((i_grp, j_grp))

    # 难度进一步提升 K_V:
    kv_grp_rel = set()
    v_grp_2_k_grp_list = defaultdict(list)
    # K
    for i_b, i_e in all_entities:
        # V
        for j_b, j_e in all_entities:
            if (i_b, j_b) in doc_relations['K_V_b'] and (i_e, j_e) in doc_relations["K_V_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'K_V', j_b, j_e, i_str, j_str))
                # 节点
                i_k_grp = be_2_k_grp[(i_b, i_e)]
                j_v_grp = be_2_v_grp[(j_b, j_e)]
                kv_grp_rel.add((i_k_grp, j_v_grp))
                v_grp_2_k_grp_list[j_v_grp].append(i_k_grp)

    # 最难级别 K_BRO
    # 从任务数据上来看, 一个value只能属于一个 kv 关系中;
    # 简单版本: 如果本身没有直接key， 那就算了, 当做错过; 若有, 找到兄弟所有的直接key_grp，按照票数选最高的key_grp 加入候选,
    # todo: 复杂版本: 找到所有兄弟潜在的key后，再predict 一遍
    # 来自兄弟节点的推理结果 这块先做简单的 时间不够了
    vv_grp_rel_bro = set()
    v_grp_2_k_grp_list_bro = defaultdict(list)
    ordered_v_be = sorted(all_entities.keys(), key=lambda x: x[0])
    # V
    for i_b, i_e in ordered_v_be:
        # V
        bro_grps = Counter()
        i_v_grp = be_2_v_grp[(i_b, i_e)]
        for j_b, j_e in ordered_v_be:
            if j_b >= i_b:  # i 需要为后来的节点
                continue
            if (i_b, j_b) in doc_relations['V_BRO_b'] and (i_e, j_e) in doc_relations["V_BRO_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'V_BRO', j_b, j_e, i_str, j_str))
                # 节点
                j_v_grp = be_2_v_grp[(j_b, j_e)]
                vv_grp_rel_bro.add((i_v_grp, j_v_grp))
                # 加入节点
                for j_v_k_grp in v_grp_2_k_grp_list[j_v_grp]:
                    bro_grps[j_v_k_grp] += 1
        # 最大
        if len(bro_grps) > 0:
            candidate_k_grp = bro_grps.most_common()[0][0]
            # 先不处理看看 ? 好像这样的做法影响了正常结果 得不偿失 ?
            v_grp_2_k_grp_list_bro[i_v_grp].append(candidate_k_grp)

    # 好了 边都齐了
    # 开始组装结果吧
    # 一个关系对:
    # [[k_grp1, k_grp2, ...], v_grp]
    # 一个关系对的字面值:
    # 先k grp 拿到值  然后 v_grp 拿到值, 然偶k组内拿到值, 最后 组装所有的值
    k_grp_strs = [pack_grp_str(k_grp_spans, tokenizer, doc_input_ids) for k_grp_spans in k_grps]
    v_grp_strs = [pack_grp_str(v_grp_spans, tokenizer, doc_input_ids) for v_grp_spans in v_grps]

    rel_infos = []
    # 针对每个V 提取 KV关系
    for v_grp, k_grps_list in v_grp_2_k_grp_list.items():
        # v, k_grps(所有信息), k_list, k_list_from_bro, k_k
        v_grp_keys = collect_v_grp_ks(v_grp, v_grps, k_grps, k_grps_list, v_grp_2_k_grp_list_bro[v_grp], kk_grp_rel)
        rel_info = pack_v_grp_rel_info(v_grp, v_grp_keys, v_grp_strs, k_grp_strs)
        rel_infos.append(rel_info)
    return rel_infos, all_raw_relations, k_grps, v_grps


def collect_doc_rel_results(doc_entities, doc_relations, tokenizer, doc_input_ids):
    """
        特定任务配置特定的关系组装
        return rel_infos, all_raw_relations, k_grps, v_grps
    """
    # 开始组装关系了
    # K_GRP: k k 循环 然后判断, 有的话就加入关系元组中
    # K-> 1, V -> 2
    # K_GRP
    all_raw_relations = set()
    k_grps = []
    be_2_k_grp = {}
    k_grp_di_g = nx.DiGraph()
    for i_b, i_e in doc_entities['KEY']:
        # 避免有孤立的点
        k_grp_di_g.add_node((i_b, i_e))
        for j_b, j_e in doc_entities['KEY']:
            # 这两个key span 是否有K_GRP 关系?
            if (i_b, j_b) in doc_relations['K_GRP_b'] and (i_e, j_e) in doc_relations["K_GRP_e"]:
                all_raw_relations.add((i_b, i_e, 'K_GRP', j_b, j_e))
                # 节点
                k_grp_di_g.add_edge((i_b, i_e), (j_b, j_e))
    # 组织 k_grp
    k_grp_g = k_grp_di_g.to_undirected()
    for node_sets in nx.connected_components(k_grp_g):
        # 直接按照offset 排序 香得很 先不用 边来排了
        nodes = sorted(list(node_sets), key=lambda x: x[0])
        k_grp_id = len(k_grps)
        k_grps.append(nodes)
        for node in nodes:
            be_2_k_grp[node] = k_grp_id

    # V_GRP 类似
    v_grps = []
    be_2_v_grp = {}
    v_grp_di_g = nx.DiGraph()
    # 2 => V
    for i_b, i_e in doc_entities['VALUE']:
        # 避免有孤立的点
        v_grp_di_g.add_node((i_b, i_e))
        for j_b, j_e in doc_entities['VALUE']:
            # 这两个key span 是否有K_GRP 关系?
            if (i_b, j_b) in doc_relations['V_GRP_b'] and (i_e, j_e) in doc_relations["V_GRP_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'V_GRP', j_b, j_e, i_str, j_str))
                # 节点
                v_grp_di_g.add_edge((i_b, i_e), (j_b, j_e))

    # 组织 v_grp
    v_grp_g = v_grp_di_g.to_undirected()
    for node_sets in nx.connected_components(v_grp_g):
        # 直接按照offset 排序 香得很 先不用 边来排了
        nodes = sorted(list(node_sets), key=lambda x: x[0])
        v_grp_id = len(v_grps)
        v_grps.append(nodes)
        for node in nodes:
            be_2_v_grp[node] = v_grp_id

    # 难度提升 K_K:  K1\tK2 字样
    kk_grp_rel = set()
    for i_b, i_e in doc_entities['KEY']:
        for j_b, j_e in doc_entities['KEY']:
            if (i_b, j_b) in doc_relations['K_K_b'] and (i_e, j_e) in doc_relations["K_K_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'K_K', j_b, j_e, i_str, j_str))
                # 节点
                i_grp = be_2_k_grp[(i_b, i_e)]
                j_grp = be_2_k_grp[(j_b, j_e)]
                kk_grp_rel.add((i_grp, j_grp))

    # 难度进一步提升 K_V:
    kv_grp_rel = set()
    v_grp_2_k_grp_list = defaultdict(list)
    # K
    for i_b, i_e in doc_entities['KEY']:
        # V
        for j_b, j_e in doc_entities['VALUE']:
            if (i_b, j_b) in doc_relations['K_V_b'] and (i_e, j_e) in doc_relations["K_V_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'K_V', j_b, j_e, i_str, j_str))
                # 节点
                i_k_grp = be_2_k_grp[(i_b, i_e)]
                j_v_grp = be_2_v_grp[(j_b, j_e)]
                kv_grp_rel.add((i_k_grp, j_v_grp))
                v_grp_2_k_grp_list[j_v_grp].append(i_k_grp)

    # 最难级别 K_BRO
    # 从任务数据上来看, 一个value只能属于一个 kv 关系中;
    # 简单版本: 如果本身没有直接key， 那就算了, 当做错过; 若有, 找到兄弟所有的直接key_grp，按照票数选最高的key_grp 加入候选,
    # todo: 复杂版本: 找到所有兄弟潜在的key后，再predict 一遍
    # 来自兄弟节点的推理结果 这块先做简单的 时间不够了
    vv_grp_rel_bro = set()
    v_grp_2_k_grp_list_bro = defaultdict(list)
    ordered_v_be = sorted(doc_entities['VALUE'].keys(), key=lambda x: x[0])
    # V
    for i_b, i_e in ordered_v_be:
        # V
        bro_grps = Counter()
        i_v_grp = be_2_v_grp[(i_b, i_e)]
        for j_b, j_e in ordered_v_be:
            if j_b >= i_b:  # i 需要为后来的节点
                continue
            if (i_b, j_b) in doc_relations['V_BRO_b'] and (i_e, j_e) in doc_relations["V_BRO_e"]:
                i_str = tokenizer.decode(doc_input_ids[i_b:i_e + 1], skip_special_tokens=True)
                j_str = tokenizer.decode(doc_input_ids[j_b:j_e + 1], skip_special_tokens=True)
                all_raw_relations.add((i_b, i_e, 'V_BRO', j_b, j_e, i_str, j_str))
                # 节点
                j_v_grp = be_2_v_grp[(j_b, j_e)]
                vv_grp_rel_bro.add((i_v_grp, j_v_grp))
                # 加入节点
                for j_v_k_grp in v_grp_2_k_grp_list[j_v_grp]:
                    bro_grps[j_v_k_grp] += 1
        # 最大
        if len(bro_grps) > 0:
            candidate_k_grp = bro_grps.most_common()[0][0]
            # 先不处理看看 ? 好像这样的做法影响了正常结果 得不偿失 ?
            v_grp_2_k_grp_list_bro[i_v_grp].append(candidate_k_grp)

    # 好了 边都齐了
    # 开始组装结果吧
    # 一个关系对:
    # [[k_grp1, k_grp2, ...], v_grp]
    # 一个关系对的字面值:
    # 先k grp 拿到值  然后 v_grp 拿到值, 然偶k组内拿到值, 最后 组装所有的值
    k_grp_strs = [pack_grp_str(k_grp_spans, tokenizer, doc_input_ids) for k_grp_spans in k_grps]
    v_grp_strs = [pack_grp_str(v_grp_spans, tokenizer, doc_input_ids) for v_grp_spans in v_grps]

    rel_infos = []
    # 针对每个V 提取 KV关系
    for v_grp, k_grps_list in v_grp_2_k_grp_list.items():
        # v, k_grps(所有信息), k_list, k_list_from_bro, k_k
        v_grp_keys = collect_v_grp_ks(v_grp, v_grps, k_grps, k_grps_list, v_grp_2_k_grp_list_bro[v_grp], kk_grp_rel)
        rel_info = pack_v_grp_rel_info(v_grp, v_grp_keys, v_grp_strs, k_grp_strs)
        rel_infos.append(rel_info)
    return rel_infos, all_raw_relations, k_grps, v_grps


def pack_v_grp_rel_info(v_grp, v_grp_keys, v_grp_strs, k_grp_strs):
    rel = [v_grp_keys, v_grp]
    key_strs = [k_grp_strs[k_grp] for k_grp in v_grp_keys]
    value_str = v_grp_strs[v_grp]
    rel_strs = [key_strs, value_str]
    return [rel, rel_strs]


def in_the_same_seg(i_b, i_e, j_b, j_e, size, step):
    min_b = min(i_b, j_b)

    this_seg_b = min_b // step * step
    this_seg_e = this_seg_b + size

    if i_b >= this_seg_b and i_b <= this_seg_e:
        if i_e >= this_seg_b and i_e <= this_seg_e:
            if j_b >= this_seg_b and j_b <= this_seg_e:
                if j_e >= this_seg_b and j_e <= this_seg_e:
                    return True
    return False


def collect_v_grp_ks(v_grp, v_grps, k_grps, k_grps_list, this_v_grp_2_k_grp_list_bro, kk_grp_rel, size=510, step=255):
    """
        组装一个v的key 关系 全场最复杂...
        当前 bro 推算的 k 只有 1个;
    """
    # 画图
    k_di_g = nx.DiGraph()
    for i_k_grp in k_grps_list:
        k_di_g.add_node(i_k_grp)
        for j_k_grp in k_grps_list:
            k_di_g.add_node(j_k_grp)
            if (i_k_grp, j_k_grp) in kk_grp_rel:
                k_di_g.add_edge(i_k_grp, j_k_grp)

    k_g = k_di_g.to_undirected()
    sub_ks_g = sorted(list(nx.connected_components(k_g)), key=lambda x: len(x), reverse=True)

    # 先用规则合并所有 ... 实在太麻烦了...
    direct_keys = {}
    for index, sub_ks_nodes in enumerate(sub_ks_g):
        # 出度 排序, 出度 一样 就 看offset:
        sub_k_grp_degrees = k_di_g.out_degree(sub_ks_nodes)
        direct_keys.update(sub_k_grp_degrees)

    # todo 这里面排序其实还有更复杂的 做法，  同出度的节点, 按照方位排, 现在先offset 大的在前面
    direct_key_and_degrees = sorted(direct_keys.items(), key=lambda x: (x[1], k_grps[x[0]][0][0]), reverse=True)
    # 同一个degree 那就

    # 处理 bro的节点 #规则
    # bro来的key 不在 直接key内，且和value不在一个片段内, 就加入到当前的列表中的最后一个来;
    result_keys = [k for k, d in direct_key_and_degrees]
    if len(this_v_grp_2_k_grp_list_bro) != 0:
        bro_k = this_v_grp_2_k_grp_list_bro[0]
        bro_b, bro_e = k_grps[bro_k][0]
        v_b, v_e = v_grps[v_grp][0]
        if bro_k not in k_grps_list and not in_the_same_seg(bro_b, bro_e, v_b, v_e, size, step):
            result_keys.append(bro_k)
    return result_keys


def pack_grp_str(grp_spans, tokenizer, doc_input_ids):
    # grp 有可能有重叠, 没有事, 融合一下
    # grp 内节点本身有序, 要求有序
    num_nodes = len(grp_spans)
    i = 0
    spans = []
    while i < num_nodes:
        i_b, i_e = grp_spans[i]
        j = i + 1
        while j < num_nodes:
            j_b, j_e = grp_spans[j]
            # 有交集 而不是包含
            if j_b <= i_e:
                if j_e > i_e:
                    i_e = j_e
                    j += 1
                else:
                    j += 1
                    continue
            else:
                break
        i = j
        spans.append((i_b, i_e))
    # 组装
    grp_strs = []
    for span in spans:
        span_input_ids = doc_input_ids[span[0]:span[1] + 1]
        span_str = tokenizer.decode(span_input_ids, skip_special_tokens=True).strip()
        grp_strs.append(span_str)
    return "".join(grp_strs)


def collct_result_from_logits(logits, offsets):
    """
        "GP result"
        logits: [batch, num_entity_types+num_rel_types*2, seq, seq]
    """
    # 先直接存
    result = [defaultdict(list) for _ in range(logits.shape[0])]
    # 要考虑offset 做对应的平移
    for b, l, start, end in zip(*np.where(logits > 0)):
        if start == 0:
            start = 1
        if end >= logits.shape[-1]-1:
            end == logits.shape[-1] - 2
        # if start == 0:
        #     start = 1
        # if end == logits.shape[-1]:
        #     end -= 1
        result[b][l].append([start - 1 + offsets[b], end - 1 + offsets[b]])
    return result


def custom_pred(trainer, test_dataset, tokenizer):
    # 搞个dataloader
    #
    dataloader = trainer.get_eval_dataloader(test_dataset)

    prediction_loss_only = False
    model = trainer._wrap_model(trainer.model, training=False)
    batch_size = dataloader.batch_size

    logger.info(f"***** Running custom pred *****")
    if isinstance(dataloader.dataset, collections.abc.Sized):
        logger.info(f"  Num examples = {trainer.num_examples(dataloader)}")
    else:
        logger.info("  Num examples: Unknown")
    logger.info(f"  Batch size = {batch_size}")

    model.eval()

    trainer.callback_handler.eval_dataloader = dataloader

    result_collections = defaultdict(list)

    observed_num_examples = 0
    # Main evaluation loop
    step = 0
    for inputs in tqdm.tqdm(dataloader):
        step += 1
        if step % 20 == 0:
            print("step:", step)
        # Update the observed num examples
        observed_batch_size = find_batch_size(inputs)
        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size
            # For batch samplers, batch_size is not known by the dataloader in advance.
            if batch_size is None:
                batch_size = observed_batch_size

        # Prediction step
        loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only, ignore_keys=None)

        # label_result = collct_result_from_logits(labels.cpu().numpy(), inputs['in_doc_token_offset'])
        result = collct_result_from_logits(logits[1].cpu().numpy(), inputs['in_doc_token_offset'])

        for x in zip(inputs['doc_ids'], inputs['in_doc_token_offset'], result):
            doc_id, offset, seg_result = x
            result_collections[doc_id].append((offset, seg_result))

    # print(result_collections)
    entity_result = collect_entity_result(result_collections,test_dataset, tokenizer)
    # print(entity_result)
    # label_entity_result = collect_entity_result(label_result_collections, eval_dataset, tokenizer)
    # print(label_entity_result)
    # 关系组装
    rel_result = collect_rel_result(result_collections, test_dataset, tokenizer)
    # dump_eval(rel_result, "eval_rel_debug.jsonl")
    return entity_result, rel_result


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=len(label2ids_span_gp),
        # finetuning_task=data_args.task_name,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.trained_model_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
    )

    image_base_path = os.path.join(data_args.data_dir, 'images')

    if training_args.do_predict:
        # 指定路径加载test
        test_ids_order = [l.strip() for l in open(os.path.join(data_args.data_dir, "test_ids.txt")).read().split(
            "\n") if l.strip() != ""]
        # test_ids_order = test_ids_order[-4:]
        real_ocr_path = os.path.join(data_args.data_dir, "task1_recg_text")
        test_dataset = t1_test_dataset_gp(data_args, test_ids_order, tokenizer, image_base_path,
                                          real_ocr_path = real_ocr_path)
        #                                  real_ocr_path=real_ocr_path, max_records=10)
        logger.info(f"test_dataset loaded: len: {len(test_dataset)}")

    encoder = LayoutLMv3Model(config=config)

    model = GlobalPointerRelModel(encoder, config, ent_type_size=len(label2ids_span_gp),
                                  rel_type_size=len(rel_types_sp), fixed_text_len=512, RoPE=True,
                                  RoPE_dim=2)

    # eval 1 model
    # trained_model_file = "/home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230313/checkpoint-500/pytorch_model.bin"
    # trained_model_file = "/home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230314/checkpoint-800/pytorch_model.bin"
    # trained_model_file = "/home/ana/data2/projects/layoutlm2023/unilm/layoutlmv3/run_t1_rel_gp_output_dir.20230315/checkpoint-800/pytorch_model.bin"
    trained_model_file = os.path.join(model_args.trained_model_path, "pytorch_model.bin")
    logger.info("weights loaded from {}".format(trained_model_file))
    state_dict = torch.load(trained_model_file)
    model.load_state_dict(state_dict, strict=False)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    # Data collator for global pointer
    # 核心的label 处理工作在这里
    data_collator = DataCollatorForKeyValueExtractionGp(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
        num_labels=len(label2ids_span_gp),
        num_rel_labels=len(rel_types_sp),
        fixed_text_length=512
    )

    metric_helper = MetricsCalculator()
    # Initialize our Trainer
    # train_dataset=train_dataset if training_args.do_train else None,
    # eval_dataset=eval_dataset if training_args.do_eval else None,
    # compute_metrics = compute_metrics,
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 先跑起来
    if training_args.do_predict:
        logger.info("*** Predict End2End***")
        # evaluation_loop
        entity_result, rel_result = custom_pred(trainer, test_dataset, tokenizer)
        # dump_rel_result('test.result', rel_result)
        dump_rel_result('test.result.new_model_1000.0331.re2.json', rel_result)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
