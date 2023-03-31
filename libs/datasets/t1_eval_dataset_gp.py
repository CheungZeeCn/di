"""
针对 task 1 的数据处理函数
单独做了一个 eval的 dataset  怕混淆
历史遗留 可以忽略
"""

import os
import json
import copy
from copy import deepcopy
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import itertools
import logging

from collections import defaultdict
from shapely.geometry import Polygon

from layoutlmft.data.image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic

label2ids_span_gp = {
    "BACKGROUND": 0,
    "KEY": 1,
    "VALUE": 2,
}
label2ids_bbox_gp = {
    "BACKGROUND": 0,
    "KEY": 1,
    "VALUE": 2,
    "KV-MIXED": 3,
}
# 拆分到span后 关系类型(先做单向关系, 逆关系可以提升召回):
rel_types_sp = {
    # 同一个key内， 按顺序
    "K_GRP": 0,
    # 同一组key内， 按顺序
    "K_K": 1,
    # k, v
    "K_V": 2,
    # 同一个 value 列表内, 按顺序
    "V_GRP": 3,
    # 兄弟 value
    "V_BRO": 4
}


def c_offset_to_t_offset(offsets_mapping, x):
    # todo 二分可提速
    for i, i_c_offset in enumerate(offsets_mapping):
        if x >= i_c_offset[0] and x < i_c_offset[1]:
            return i
    raise ValueError(f"x should < len(offset_mappings), but {x} >= {len(offsets_mapping)}")


def get_image_wh_by_ocr(data):
    x_values = []
    y_values = []
    for kv_pair in data["KV-Pairs"]:
        for value in kv_pair['Values']:
            x_values += [coord_v for i, coord_v in enumerate(value['Coord']) if i % 2 == 0]
            y_values += [coord_v for i, coord_v in enumerate(value['Coord']) if i % 2 == 1]
    for value in data["Backgrounds"]:
        x_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 0]
        y_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 1]
    for key, values in data["Keys"].items():
        for value in values:
            x_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 0]
            y_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 1]
    min_x = max(0, min(x_values))
    min_y = max(0, min(y_values))
    max_x = max(x_values)
    max_y = max(y_values)
    return min_x, min_y, max_x, max_y


def get_image_wh_by_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
    return 0, 0, img.size[0], img.size[1]


def fix_coord_error(coord):
    """
        简单的位置修正, 副作用是会改动coord原来的数值
    """
    coord = list(coord)
    x_values = []
    y_values = []
    x_values += [coord_v for i, coord_v in enumerate(coord) if i % 2 == 0]
    y_values += [coord_v for i, coord_v in enumerate(coord) if i % 2 == 1]
    min_x = max(0, min(x_values))
    min_y = max(0, min(y_values))
    max_x = max(max(x_values), min_x + 1)
    max_y = max(max(y_values), min_y + 1)
    coord = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
    coord = tuple(coord)
    return coord


def load_pp_ocr_data(dir_path):
    ret_data = defaultdict(dict)
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            file_id = file.replace(".txt", "")
            with open(os.path.join(dir_path, file)) as f:
                data = f.read().split("\n")
                for line in data:
                    line = line.strip()
                    if line == "":
                        continue
                    line_obj = json.loads(line)
                    coords = tuple(itertools.chain(*line_obj[0]))
                    coords = fix_coord_error(coords)
                    text = line_obj[1][0]
                    ret_data[file_id][coords] = text
    return ret_data


def to_4_points(flat_coord):
    return (flat_coord[0:2], flat_coord[2:4], flat_coord[4:6], flat_coord[6:8])


def pick_2_points(flat_coord):
    return flat_coord[0:2] + flat_coord[4:6]


def order_by_tbyx_coord(coords, th=20):
    """
	ocr_info: a list of dict, which contains bbox information([x1, y1, x2, y2])
	th: threshold of the position threshold
	"""
    res = sorted(coords, key=lambda r: (r[1], r[0]))  # sort using y1 first and then x1
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            # restore the order using the
            if abs(res[j + 1][1] - res[j][1]) < th and \
                    (res[j + 1][0] < res[j][0]):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res


class t1_eval_dataset_gp(Dataset):
    """
        ner 和关系抽取的 dataset
        eval 时候用
    """
    def __init__(
            self,
            args,
            tokenizer,
            mode,
            image_path,
            max_records=None,
            real_ocr_path=None,
            json_file=None,
            ids_order=None
    ):
        self.args = args
        self.mode = mode
        self.cur_la = args.language
        self.tokenizer = tokenizer
        self.ner_label2ids_span = label2ids_span_gp
        self.ner_label2ids_bbox = label2ids_bbox_gp
        self.rel_label2ids = rel_types_sp
        self.image_path = image_path
        self.ids_order = ids_order
        # 放一些数据方便后续回溯
        self.data_store = {}
        self.total_data = None
        self.total_inputs = None
        self.doc_id_to_index = {}
        self.data = None

        self.common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, interpolation=args.train_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.5, 0.5, 0.5)),
                std=torch.tensor((0.5, 0.5, 0.5)))
        ])

        if json_file is None:
            json_file = os.path.join(args.data_dir, "{}.json".format('train' if mode == 'train' else 'test'))
        data_file = json.load(open(json_file, 'r'))
        if real_ocr_path is not None:
            real_ocr_data = load_pp_ocr_data(real_ocr_path)
        else:
            real_ocr_data = None

        self.feature = self.load_data(data_file, max_records, real_ocr_data=real_ocr_data)
        logging.info(
            f"loading file {json_file}, get {len(data_file)} examples, get {len(self.feature['input_ids'])} records")

    def load_data(
            self,
            data_file,
            max_records=None,
            real_ocr_data=None
    ):
        # re-org data format
        # 这里的lines 其实就是里面的bbox的内容
        # todo: 添加关系数据的处理
        total_data = {
            "id": [],
            "lines": [],
            "bboxes": [],
            "ner_tags": [],
            "image_path": [],
            # 每个box的span info # token label 包含信息
            "doc_spans": [],

            # "token_tags": [],
            # list of dict? 等做到关系部分再去决定怎么处理
            "rel_labels": []
        }

        num_records = len(data_file)
        if max_records is not None:
            num_records = min(max_records, num_records)

        if self.ids_order is None:
            file_ids = sorted(data_file.keys())
        else:
            file_ids = self.ids_order

        for i in range(num_records):
            file_id = file_ids[i]
            # 数据预处理
            i_data = data_file[file_id]
            if real_ocr_data is not None and file_id in real_ocr_data:
                i_real_ocr_data = real_ocr_data[file_id]
            else:
                i_real_ocr_data = None

            i_preprocessed_data = self.prepare_image_data(file_id, i_data, i_real_ocr_data)
            # 暂存一下
            self.data_store[file_id] = {'preprocessed_data': i_preprocessed_data}
            # todo: 处理关系部分数据;
            i_merged_key_maps, i_merged_span_info, i_merged_rel_info, i_merged_bbox_info, \
            i_merged_rel_labels = i_preprocessed_data

            min_x, min_y, max_x, max_y = get_image_wh_by_image(os.path.join(self.image_path, file_id))
            width = max_x - min_x
            height = max_y - min_y

            # token tags 相对麻烦一些; 不过当前的任务是没有重叠的(有重叠的要换一种记录方式)
            cur_doc_lines, cur_doc_bboxes, cur_doc_ner_tags, cur_doc_image_path, cur_doc_spans = [], [], [], [], []

            for j in range(len(i_merged_bbox_info)):
                cur_item = i_merged_bbox_info[j]
                cur_doc_lines.append(cur_item['text'])
                # print(cur_item['text'], file_id)
                cur_doc_bboxes.append(self.box_norm(pick_2_points(cur_item['coord']), width=width, height=height))
                cur_doc_ner_tags.append(cur_item['type'])
                # 每个box的span 列表
                cur_doc_spans.append(cur_item['spans'])

            total_data['id'] += [file_id, ]
            total_data['lines'] += [cur_doc_lines, ]
            total_data['bboxes'] += [cur_doc_bboxes, ]
            total_data['ner_tags'] += [cur_doc_ner_tags, ]
            total_data['doc_spans'] += [cur_doc_spans, ]
            total_data['image_path'] += [file_id, ]
            total_data['rel_labels'] += [i_merged_rel_labels, ]

        self.total_data = total_data
        # tokenize text and get bbox/label
        total_input_ids, total_bboxs, total_bbox_label_ids, total_span_label_ids = [], [], [], []
        total_token_len, total_span_rel_labels = [], []
        total_bbox_ids = []
        # doc -> span_id -> [offset_i, j] (未加特殊标记时)
        total_span_offsets = []

        # 文档 i 的完全体
        for i in range(len(total_data['lines'])):
            cur_doc_id = total_data['id'][i]
            self.doc_id_to_index[cur_doc_id] = i
            cur_doc_input_ids, cur_doc_bboxs, cur_doc_bbox_labels, cur_doc_span_labels = [], [], [], []
            cur_doc_token_len = 0
            now_token_offset = 0
            # span_id => 全文的token offset, 未经过处理;
            cur_doc_span_offset_dict = {}

            # bbox
            for j in range(len(total_data['lines'][i])):
                line_encoded = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False,
                                              return_attention_mask=False, return_offsets_mapping=True)
                cur_input_ids = line_encoded['input_ids']
                if len(cur_input_ids) == 0:
                    continue
                # 转大写
                cur_bbox_label = total_data['ner_tags'][i][j].upper()

                cur_offsets_mapping = line_encoded['offset_mapping']
                # ignore o label
                cur_bbox_span_labels = [-100] * len(cur_input_ids)
                # 要有映射关系才能拿到span对应的位置，才能给token 赋值label
                for span_info in total_data['doc_spans'][i][j]:
                    char_offset_i = span_info['offset']
                    char_offset_j = span_info['offset'] + span_info['length'] - 1
                    span_type = span_info['type']
                    # debug
                    # if cur_doc_id == '100.jpg' and span_info['id'] == 18:
                    #     print("xx")
                    #     pass
                    try:
                        token_offset_i = c_offset_to_t_offset(cur_offsets_mapping, char_offset_i)
                        token_offset_j = c_offset_to_t_offset(cur_offsets_mapping, char_offset_j)
                    except Exception as e:
                        print(cur_offsets_mapping)
                        print(span_info)
                        print(total_data['doc_spans'][i][j])
                        print(total_data['id'][i])
                        raise e

                    for k in range(token_offset_i + 1, token_offset_j + 1):
                        cur_bbox_span_labels[k] = self.ner_label2ids_span[span_type] * 2
                    # 找到原因了。。。
                    cur_bbox_span_labels[token_offset_i] = self.ner_label2ids_span[span_type] * 2 + 1

                    # 闭区间
                    cur_doc_span_offset_dict[span_info['id']] = [now_token_offset + token_offset_i,
                                                                 now_token_offset + token_offset_j]

                # 这里的数据格式里面 同一个bbox 只能有一个 label;
                # gp 没有O标签也没有关系, 可以忽略这部分
                if cur_bbox_label == 'OTHER':
                    # ignore o label
                    cur_bbox_labels = [-100] * len(cur_input_ids)
                else:
                    # 为了识别出边界
                    cur_bbox_labels = [self.ner_label2ids_bbox[cur_bbox_label] * 2] * len(cur_input_ids)
                    # 左bbox边界 需要配套改 collect gp label 函数
                    cur_bbox_labels[0] = cur_bbox_labels[0] + 1
                assert len(cur_input_ids) == len([total_data['bboxes'][i][j]] * len(cur_input_ids)) == len(
                    cur_bbox_labels) == len(cur_bbox_span_labels)
                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                cur_doc_bbox_labels += cur_bbox_labels
                cur_doc_span_labels += cur_bbox_span_labels
                now_token_offset += len(cur_input_ids)

            cur_doc_token_len = now_token_offset

            assert len(cur_doc_input_ids) == len(cur_doc_bboxs) == len(cur_doc_bbox_labels) == len(cur_doc_span_labels), \
                f"{len(cur_doc_input_ids)} == {len(cur_doc_bboxs)} == {len(cur_doc_bbox_labels)} == {len(cur_doc_span_labels)}"
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            total_bbox_label_ids.append(cur_doc_bbox_labels)
            total_span_label_ids.append(cur_doc_span_labels)
            # 所有token的长度
            total_token_len.append(cur_doc_token_len)

            # relations
            # span之间的关系信息 + bbox处理过程中手机的span的offset信息; 简简单单有没有..
            ## 输入格式: merged_rel_labels[rel_types_sp["K_GRP"]].append([keys[i], keys[j]])
            num_rel_types = len(rel_types_sp)
            cur_doc_span_rel_labels = np.zeros([num_rel_types * 2, cur_doc_token_len, cur_doc_token_len])
            # 直接初始化一个numpy矩阵吧
            for j_type, rel_labels in enumerate(total_data['rel_labels'][i]):
                # 每个关系对
                for h_span_id, t_span_id in rel_labels:
                    h_offset_b, h_offset_e = cur_doc_span_offset_dict[h_span_id]
                    t_offset_b, t_offset_e = cur_doc_span_offset_dict[t_span_id]
                    # head
                    cur_doc_span_rel_labels[j_type][h_offset_b][t_offset_b] = 1
                    # tail
                    cur_doc_span_rel_labels[j_type + num_rel_types][h_offset_e][t_offset_e] = 1
                    # debug
                    # try:
                    #     cur_doc_span_rel_labels[j_type + num_rel_types][h_offset_e][t_offset_e] = 1
                    # except Exception as e:
                    #     print(f"[{j_type} + {num_rel_types}][{h_offset_e}][{t_offset_e}]")
                    #     raise e
            total_span_rel_labels.append(cur_doc_span_rel_labels)
        assert len(total_input_ids) == len(total_bboxs) == len(total_bbox_label_ids) == len(
            total_span_label_ids) == len(total_span_rel_labels)

        self.total_inputs = {
            "total_input_ids": total_input_ids,
            "total_bboxs": total_bboxs,
            "total_bbox_label_ids": total_bbox_label_ids,
            "total_span_label_ids": total_span_label_ids,
            "total_token_len": total_token_len,
            "total_span_rel_labels": total_span_rel_labels
        }

        """
            到目前为止，在eval场景下需要用到的语料都有

            total_input_ids, total_span_rel_labels, 
            total_bboxs, total_bbox_label_ids, total_span_label_ids

        """

        # todo: 这里做关系处理 的截断 会 比较复杂一些 DONE; total_span_rel_labels 里面存放着我们关心的关系信息;  i, i+num_rel_types 为对应关系的头和尾
        # split text to several slices because of over-length
        # 增加 step 变量，控制滑动的步长

        step_size = 255
        max_content_len = 510

        input_ids, bboxs, bbox_labels, span_labels = [], [], [], []
        segment_ids, position_ids = [], []
        gp_bbox_labels = []
        gp_span_labels = []
        # 关系的labels 其实十分容易处理, 取对应total_span_rel_labels片段即可
        gp_rel_labels = []

        in_doc_token_offset = []
        doc_ids = []

        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
            i_doc_span_rel_labels = total_span_rel_labels[i]
            while start < len(total_input_ids[i]):
                # DONE: 这部分的截断可以更灵活一些，取box的边界再截断
                end = min(start + max_content_len, len(total_input_ids[i]))
                # 开始判断 end 是否合适, 关注是否文档边界， 单个bbox是否超过510
                if end == len(total_input_ids[i]):
                    # 到文档边界了， 所以不用处理
                    pass
                else:
                    # 寻找bbox边界
                    while total_bboxs[i][end] == total_bboxs[i][end - 1] and end != start:
                        end -= 1
                    # 这时候 end 可能为bbox边界 或者 start(box超长的时候)
                    if end <= start:
                        # box 超长只能截断, 一般不会出现这个情况
                        end = start + max_content_len

                input_ids.append(
                    [self.tokenizer.cls_token_id] + total_input_ids[i][start: end] + [self.tokenizer.sep_token_id])
                bboxs.append([[0, 0, 0, 0]] + total_bboxs[i][start: end] + [[1000, 1000, 1000, 1000]])

                cur_bbox_labels = [-100] + total_bbox_label_ids[i][start: end] + [-100]
                cur_span_labels = [-100] + total_span_label_ids[i][start: end] + [-100]

                in_doc_token_offset.append(start)
                doc_ids.append(total_data['id'][i])

                # 超简单的处理
                # cur_gp_rel_span_labels = np.zeros([num_rel_types*2, end-start+2, end-start+2], dtype=np.int64)
                cur_gp_rel_span_labels = np.zeros([num_rel_types * 2, end - start + 2, end - start + 2])
                cur_gp_rel_span_labels[:, 1:-1, 1:-1] = total_span_rel_labels[i][:, start:end, start:end]

                bbox_labels.append(cur_bbox_labels)
                span_labels.append(cur_span_labels)

                cur_segment_ids = self.get_segment_ids(bboxs[-1])
                cur_position_ids = self.get_position_ids(cur_segment_ids)

                cur_gp_bbox_labels = self.collect_gp_labels(cur_bbox_labels)
                cur_gp_span_labels = self.collect_gp_labels(cur_span_labels)

                segment_ids.append(cur_segment_ids)
                position_ids.append(cur_position_ids)
                image_path.append(os.path.join(self.args.data_dir, "images", total_data['image_path'][i]))

                gp_bbox_labels.append(cur_gp_bbox_labels)
                gp_span_labels.append(cur_gp_span_labels)
                gp_rel_labels.append(cur_gp_rel_span_labels)

                #
                if end == len(total_input_ids[i]):
                    # 边界 后续会被忽略
                    next_start = end
                else:
                    # 滑动窗口
                    next_start = min(start + step_size, end)
                    if next_start == end:
                        # 边界, 不用再滑动了
                        pass
                    while total_bboxs[i][next_start] == total_bboxs[i][next_start - 1] and next_start != start:
                        next_start -= 1
                    if next_start <= start:
                        # box 超长只能截断, 一般不会出现这个情况
                        next_start = start + step_size
                # 下个片段走起
                start = next_start
                cur_iter += 1

        assert len(input_ids) == len(bboxs) == len(bbox_labels) == len(span_labels) == len(segment_ids) == len(
            position_ids) == len(gp_bbox_labels) == len(gp_span_labels) == len(gp_rel_labels)
        assert len(segment_ids) == len(image_path)

        # todo: 考虑截断处不要在一个bbox中间; done
        # 1. 长文本截断分片，无overlap(这部分在 NER 应该影响不大，但是关系抽取可能会有一定影响);
        # 2. 这里labels仅仅做了id的转化，后续还需要进一步处理, 考虑到最终位移的情况 #
        # 3. gp_labels 列表(batch_size)的列表(single record)的列表(i,j,label), [ [i, j, lable_id], [i,j,label_id]]
        #
        # 先看span level的效果?
        res = {
            'input_ids': input_ids,
            'bbox': bboxs,
            'ori_labels': span_labels,
            'labels': gp_span_labels,
            'gp_rel_labels': gp_rel_labels,
            'segment_ids': segment_ids,
            'position_ids': position_ids,
            'image_path': image_path,
            'in_doc_token_offset': in_doc_token_offset,
            'doc_ids': doc_ids
        }
        self.data = res
        return res

    def box_norm(self, box, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        x0, y0, x1, y1 = box
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)
        # 有些图片旋转的比较厉害 就没有办法咯
        if x1 < x0:
            x1 = x0 + 1
        if y1 < y0:
            y1 = y0 + 1
        assert x1 >= x0 and x0 >= 0, f"{x1}, {x0}"
        assert y1 >= y0 and y0 >= 0, f"{y1}, {y0}"
        return [x0, y0, x1, y1]

    def get_segment_ids(self, bboxs):
        segment_ids = []
        for i in range(len(bboxs)):
            if i == 0:
                segment_ids.append(0)
            else:
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids

    def get_position_ids(self, segment_ids):
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids

    def collect_gp_labels(self, cur_labels):
        """
            cur_gp_labels: 遍历cur_labels 把 != -100 的 slice的 i,j 找出来; 由于是闭合区间，所以还需要j-1
            bi， label // 2 才是真实值， 是为了区分 左边界;
        """
        cur_gp_labels = []
        i = 0
        while i < len(cur_labels):
            i_label = cur_labels[i]
            # begin
            if i_label != -100:
                for j in range(i + 1, len(cur_labels)):
                    j_label = cur_labels[j]
                    # 相同label 且 不为 B-*** 的时候
                    if (j_label // 2 == i_label // 2) and (j_label % 2 != 1):
                        j += 1
                        continue
                    else:
                        break
                cur_gp_labels.append([i, j - 1, i_label // 2])
                i = j
                continue
            else:
                # next
                i += 1
        return cur_gp_labels

    def collect_gp_labels_old(self, cur_labels):
        # cur_gp_labels: 遍历cur_labels 把 != -100 的 slice的 i,j 找出来; 由于是闭合区间，所以还需要j-1
        cur_gp_labels = []
        i = 0
        while i < len(cur_labels):
            i_label = cur_labels[i]
            # begin
            if i_label != -100:
                for j in range(i, len(cur_labels)):
                    j_label = cur_labels[j]
                    if j_label == i_label:
                        j += 1
                        continue
                    else:
                        break
                # 检查j是到了停止边界呢，还是遇见了不同;
                cur_gp_labels.append([i, j - 1, i_label])
                i = j
                continue
            else:
                # next
                i += 1
        return cur_gp_labels

    def collect_ori_image_info(self, data):
        """
            针对单个文件
            转换格式
        """
        # 原始key => 当前的bbox id 列表
        key_maps = defaultdict(list)
        # bbox 的信息
        bbox_info = {}
        rel_info = []
        id_num = 0

        for value in data["Backgrounds"]:
            coord = tuple(value['Coords'])
            coord = fix_coord_error(coord)
            text = value['Content']
            if text.strip() == "":
                continue
            type = "BACKGROUND"
            bbox_info[coord] = {"type": type, "text": text, "coord": fix_coord_error(coord), "id": id_num}
            id_num += 1

        for key, values in data["Keys"].items():
            key = int(key)
            for value in values:
                coord = tuple(value['Coords'])
                coord = fix_coord_error(coord)
                text = value['Content']
                type = 'KEY'
                bbox_info[coord] = {"type": type, "text": text, "coord": fix_coord_error(coord), "id": id_num}
                key_maps[key].append(id_num)
                id_num += 1

        # 两次遍历 第一次拿 id; 第二次填关系
        for kv_pair in data["KV-Pairs"]:
            for value in kv_pair['Values']:
                coord = tuple(value['Coord'])
                text = value['Content']
                type = 'VALUE'
                if coord not in bbox_info:
                    coord = fix_coord_error(coord)
                    bbox_info[coord] = {"type": type, "text": text, "coord": fix_coord_error(coord), "id": id_num}
                    id_num += 1
                else:
                    raise ValueError(
                        "VALUE BBOX {} properties conflict: before:{} after: {}".format(coord, bbox_info[coord],
                                                                                        {"type": type, "text": text,
                                                                                         "coord": coord, "id": id_num}))
        for kv_pair in data["KV-Pairs"]:
            key_info = kv_pair["Keys"]
            value_info = []
            for value in kv_pair['Values']:
                coord = tuple(value['Coord'])
                coord = fix_coord_error(coord)
                value_id = bbox_info[coord]["id"]
                value_info.append(value_id)
            rel_info.append([key_info, value_info])

        return key_maps, bbox_info, rel_info

    def merge_bbox_info(self, key_maps, bbox_info, rel_info, real_ocr_data=None, contrain_threshold=0.67):
        """
            如果real_ocr_data None, 就做个简单转换
            如果不是 就开始合并操作
        """
        if real_ocr_data is None:
            real_ocr_coords = list(bbox_info.keys())
        else:
            real_ocr_coords = list(real_ocr_data.keys())
        real_bbox_shapes = [Polygon(to_4_points(flat_coords)) for flat_coords in real_ocr_coords]

        # 真实ocr bbox 映射到 新的bbox信息中;
        # real2group = {}
        ori_bbox2group = {}
        # 就是合并后的grp, 直接合并的bbox 坐标 为 key, 内容为原始的ori_bbox id;
        grp_info = defaultdict(list)
        # 根据ori_bbox-> grp 的归属信息，重新组装， 并排序和给ID
        merged_bbox_info = []
        # 旧的bbox在新的数据下的表达
        spans_info = []
        ori_bbox2span = {}
        # key-> 旧的bbox->新的spans_info
        new_key_maps = {}
        # 旧的关系-> 新的 spans_info
        new_rel_info = []

        # bbox_info[coord] = {"type":type, "text": text, "coord": coord, "id":id_num}
        # 按照id排序
        ori_bbox_info = sorted(bbox_info.values(), key=lambda x: x["id"])

        for i, i_ori_info in enumerate(ori_bbox_info):
            i_coord = i_ori_info["coord"]
            i_shape = Polygon(to_4_points(i_coord))
            assert i == i_ori_info["id"] and isinstance(i_coord, tuple)
            # 看是否会被包含
            contain_flag = False
            for j, j_shape in enumerate(real_bbox_shapes):
                j_coord = real_ocr_coords[j]
                assert isinstance(j_coord, tuple)
                try:
                    if i_shape.intersection(j_shape).area / i_shape.area > contrain_threshold:
                        contain_flag = True
                        grp_info[j_coord].append(i_ori_info["id"])
                        break
                except Exception as e:
                    print(i_coord, j_coord)
                    raise e
                # if i_ori_info["id"] == 10 and i_shape.intersection(j_shape).area / i_shape.area > 0.5:
                #     print("----")
                #     print(i_ori_info, j_shape, i_shape.intersection(j_shape).area / i_shape.area)
                #     print("----")
            # 独自成grp
            if contain_flag is not True:
                grp_info[i_coord].append(i_ori_info["id"])

        # 已经合并完毕，开始排序和生成对应的新ID; 排列顺序 从左到右，从上到下
        # grp_coords = sorted(grp_info.keys(), key=lambda coord: (coord[1], coord[0]))
        grp_coords = order_by_tbyx_coord(grp_info.keys())

        # bbox_info[coord] = {"type":type, "text": text, "coord": coord, "id":id_num}
        offset = 0
        for i, i_coord in enumerate(grp_coords):
            # type, text, coord, id, length
            # spans:  [{span级别的标注信息}, {span_id, 在这里的开始，在这里的结束为止, 文本}]
            # 关系的label数据放到全局部分处理
            i_merge_bbox = {"id": i, "coord": i_coord, "text": "", "types": [], "type": None, "length": None,
                            "offset": offset,
                            "spans": []
                            }
            # grp 内排序
            i_grp_ori_ids = grp_info[i_coord]
            if len(i_grp_ori_ids) != 1:
                sorted_ori_box_in_i = sorted([(ori_id, ori_bbox_info[ori_id]) for ori_id in i_grp_ori_ids],
                                             key=lambda x: x[1]['coord'][0])

                i_grp_ori_ids = [x[0] for x in sorted_ori_box_in_i]
                # print("坐标bbox:", i_coord, "涉及合并的bbox", i_grp_ori_ids)

            in_grp_offset = 0
            for ori_id in i_grp_ori_ids:
                ori_bbox2group[ori_id] = i
                i_merge_bbox["text"] += ori_bbox_info[ori_id]["text"]
                i_merge_bbox["types"].append(ori_bbox_info[ori_id]["type"])
                new_span_info = copy.deepcopy(ori_bbox_info[ori_id])
                new_span_info["ori_id"] = new_span_info["id"]
                new_span_info["id"] = len(spans_info)
                new_span_info["offset"] = in_grp_offset
                new_span_info["bbox_id"] = i
                new_span_info["length"] = len(ori_bbox_info[ori_id]["text"])
                in_grp_offset += len(ori_bbox_info[ori_id]["text"])
                spans_info.append(new_span_info)
                ori_bbox2span[ori_id] = new_span_info["id"]
                i_merge_bbox["spans"].append(new_span_info)
                ori_bbox2group[ori_id] = i
            # length
            i_merge_bbox["length"] = len(i_merge_bbox["text"])
            offset += i_merge_bbox["length"]
            # type
            t_set = set(i_merge_bbox["types"])
            if len(t_set) == 1:
                i_merge_bbox["type"] = list(t_set)[0]
            else:
                if "KEY" in t_set and "VALUE" in t_set:
                    i_merge_bbox["type"] = "KV-MIXED"
                elif "KEY" in t_set:
                    i_merge_bbox["type"] = "KEY"
                else:
                    i_merge_bbox["type"] = "VALUE"

            merged_bbox_info.append(i_merge_bbox)
        # 直到这里，已经完成计算: merged_bbox_info, ori_bbox2group, ori_bbox_info, ori_bbox2span
        # 配合 之前的 key_maps
        for k, bbox_ids in key_maps.items():
            new_key_maps[k] = [ori_bbox2span[bbox_id] for bbox_id in bbox_ids]

        for ksvs in rel_info:
            keys = ksvs[0]
            values = ksvs[1]
            new_values = [ori_bbox2span[bbox_id] for bbox_id in values]
            new_rel_info.append([keys, new_values])

        # 训练数据， 真实输入数据
        return new_key_maps, spans_info, new_rel_info, \
               merged_bbox_info

    def collect_rel_labels(self, merged_key_maps, merged_rel_info, rel_types_sp):
        """
            关系数据
            # 每个关系类型为一个子列表
            [
                [[头span_id, 尾span_id], [...]] ,
                ...,
                []
            ]
        """
        merged_rel_labels = [list() for i in range(len(rel_types_sp))]
        # K_GRP, 0， 同一个key内的各个span 按照id 来单向的给关系;
        # 在还原的时候 就是 连通图 + 排序算法;
        for k_id, keys in merged_key_maps.items():
            for i in range(len(keys) - 1):
                for j in range(i + 1, len(keys)):
                    merged_rel_labels[rel_types_sp["K_GRP"]].append([keys[i], keys[j]])

        # keys 最后一个key -> 所有的value span 的第一个 就是兄弟节点 (暂时不考虑 一个 value 下的所有span)
        # key id -> value的 第一个 span id
        # 后续使用 value 的 V_GRP 还原成 单一的value 后， 通过兄弟节点 ，可以连接到依赖的更年长的兄弟节点的某个关系里面的最后一个key
        # 注意这里兄弟之间也是有年长年幼的，先加入的节点是年长的，兄弟关系只有单向的 年幼->年长， 找哥哥，通过哥哥找到爸爸;
        # 另外一个value 不会归属两个关系内，所以剩下的问题只是拿到所有的key后，去排除别的key，然后对自己的key排序即可;
        key_to_1st_spans = defaultdict(list)

        for keys, value_spans in merged_rel_info:
            if len(keys) == 0:
                continue

            # 最后一个key的id, 和value的第一个span，为了后续生成兄弟节点关系
            last_key_id = keys[-1]
            first_value_span_id = value_spans[0]
            key_to_1st_spans[last_key_id].append(first_value_span_id)

            # K_K 1 # \t 分割那种, 同一组key内, 相邻的key进行传递
            # 这也是有方向的那种
            for i in range(0, len(keys) - 1):
                j = i + 1
                k1_sids = merged_key_maps[keys[i]]
                k2_sids = merged_key_maps[keys[j]]
                # 仅用开头的span来做代表
                sid1 = k1_sids[0]
                sid2 = k2_sids[0]
                merged_rel_labels[rel_types_sp["K_K"]].append([sid1, sid2])
                # for sid1 in k1_sids:
                #     for sid2 in k2_sids:
                #         merged_rel_labels[rel_types_sp["K_K"]].append([sid1, sid2])

            #  K_V 2 #
            # 每一个key的第一个value 和 每一个v的第一个value;
            for k_id in keys:
                k_span_ids = merged_key_maps[k_id]
                # 每一个key 的span
                k_sid = k_span_ids[0]
                v_sid = value_spans[0]
                merged_rel_labels[rel_types_sp["K_V"]].append([k_sid, v_sid])

                # for k_sid in k_span_ids:
                #     # 每一个v span
                #     for v_sid in value_spans:
                #         # 每一个k_span 和 v_span 都有关系
                #         # 这里面就做了些关系冗余，可能到时候不大好做后处理
                #         merged_rel_labels[rel_types_sp["K_V"]].append([k_sid, v_sid])

            """
            # 做个最简单版本的
            # 最后一个key
            k_id = keys[-1]
            # 最后一个key的span_ids
            k_span_ids = merged_key_maps[k_id]
            # 最后一个key 的最后一个 span_id
            k_sid = k_span_ids[-1]
            v_sid = value_spans[0]
            merged_rel_labels[rel_types_sp["K_V"]].append([k_sid, v_sid])
            """

            #  V_GRP 3 同一个value内 按顺序
            for i in range(0, len(value_spans) - 1):
                for j in range(i + 1, len(value_spans)):
                    merged_rel_labels[rel_types_sp["V_GRP"]].append([value_spans[i], value_spans[j]])

        for key, bro_spans in key_to_1st_spans.items():
            bro_spans = sorted(bro_spans)
            for i, i_sid in enumerate(bro_spans):
                for j in range(i + 1, len(bro_spans)):
                    j_sid = bro_spans[j]
                    merged_rel_labels[rel_types_sp["V_BRO"]].append([j_sid, i_sid])

        return merged_rel_labels

    def prepare_image_data(self, i_id, i_data, i_real_ocr_data):
        """
            结合 paddleOCR 的真实输出, 构造训练数据
        """
        # t1 格式原始数据加载
        key_maps, bbox_info, rel_info = self.collect_ori_image_info(i_data)
        merged_key_maps, merged_span_info, merged_rel_info, \
        merged_bbox_info = self.merge_bbox_info(key_maps, bbox_info, rel_info, i_real_ocr_data)
        # map rel_info
        merged_rel_labels = self.collect_rel_labels(merged_key_maps, merged_rel_info, rel_types_sp)
        return merged_key_maps, merged_span_info, merged_rel_info, merged_bbox_info, merged_rel_labels

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        # attention_mask = self.feature["attention_mask"][index]
        attention_mask = [1] * len(input_ids)
        labels = self.feature["labels"][index]
        rel_labels = self.feature["gp_rel_labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        img = pil_loader(self.feature['image_path'][index])
        for_patches, _ = self.common_transform(img, augmentation=False)
        patch = self.patch_transform(for_patches)


        in_doc_token_offset = self.feature['in_doc_token_offset'][index]
        doc_ids = self.feature['doc_ids'][index]

        assert len(input_ids) == len(attention_mask) == len(bbox) == len(segment_ids), \
            ("len(input_ids):{} == len(attention_mask):{} == len(labels):{}"
             " == len(bbox):{} == len(segment_ids):{}").format(
                len(input_ids), len(attention_mask), len(bbox), len(segment_ids))

        # 临时加了一个rel_labels 这个和labels 结构不一样，但是实际上应该一样才对, 要改一下collate
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rel_labels": rel_labels,
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "images": patch,
            'in_doc_token_offset': in_doc_token_offset,
            'doc_ids': doc_ids
        }
        return res


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
