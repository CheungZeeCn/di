"""
针对 task 1 的数据处理函数
单独做了一个 eval的 dataset  怕混淆
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

from libs.datasets.image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic, pil_loader

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


def collect_file_task_info(file_id, ocr_base):
    """
        官方的格式
    """
    ocr_file = os.path.join(ocr_base, file_id+".txt")
    ret = []
    with open(ocr_file) as f:
        for l in f:
            if l.strip() == "":
                continue
            lsp = l.split(" ")
            min_x, min_y, w, h = [int(v) for v in lsp[:4]]
            max_x = min_x + w
            max_y = min_y + h
            text = " ".join(lsp[4:])
            if text.strip() == "":
                continue
            coord = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
            coord = fix_coord_error(coord)
            bbox = {'coord':coord, "text": text}
            ret.append(bbox)
    return ret

def fix_coord_error(coord):
    """
        简单的位置修正, 副作用是会改动coord原来的数值
    """
    coord = list(coord)
    x_values = [coord_v for i, coord_v in enumerate(coord) if i % 2 == 0]
    y_values = [coord_v for i, coord_v in enumerate(coord) if i % 2 == 1]
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


class t1_test_dataset_gp(Dataset):
    """
        ner 和关系抽取的 dataset
        eval 时候用
    """

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

    def gen_bbox_info(self, file_bbox_info):
        """
            如果real_ocr_data None, 就做个简单转换
            如果不是 就开始合并操作
        """
        new_bbox_info = []
        grp_info = {}
        for bbox_info in file_bbox_info:
            grp_info[bbox_info['coord']] = bbox_info

        grp_coords = order_by_tbyx_coord(grp_info.keys())

        # bbox_info[coord] = {"type":type, "text": text, "coord": coord, "id":id_num}
        offset = 0
        for i, i_coord in enumerate(grp_coords):
            # type, text, coord, id, length
            # spans:  [{span级别的标注信息}, {span_id, 在这里的开始，在这里的结束为止, 文本}]
            # 关系的label数据放到全局部分处理
            i_merge_bbox = grp_info[i_coord]
            i_merge_bbox["id"] = i
            i_merge_bbox["length"] = len(i_merge_bbox["text"])
            i_merge_bbox["offset"] = offset
            new_bbox_info.append(i_merge_bbox)

        # 训练数据， 真实输入数据
        return new_bbox_info

    def prepare_image_data(self, i_id, i_data):
        """
        """
        # t1 格式原始数据加载
        new_bbox_info = self.gen_bbox_info(i_data)
        # map rel_info
        return new_bbox_info

    def load_data(
            self,
            data_file,
    ):
        # re-org data format
        # 这里的lines 其实就是里面的bbox的内容
        # todo: 添加关系数据的处理
        total_data = {
            "id": [],
            "lines": [],
            "bboxes": [],
            "image_path": []
        }
        num_records = len(data_file)

        if self.ids_order is None:
            file_ids = sorted(data_file.keys())
        else:
            file_ids = self.ids_order

        for i in range(num_records):
            file_id = file_ids[i]
            # 数据预处理
            i_data = data_file[file_id]
            i_preprocessed_data = self.prepare_image_data(file_id, i_data)
            # 暂存一下
            self.data_store[file_id] = {'preprocessed_data': i_preprocessed_data}
            i_merged_bbox_info = i_preprocessed_data
            min_x, min_y, max_x, max_y = get_image_wh_by_image(os.path.join(self.image_path, file_id))
            width = max_x - min_x
            height = max_y - min_y

            # token tags 相对麻烦一些; 不过当前的任务是没有重叠的(有重叠的要换一种记录方式)
            cur_doc_lines, cur_doc_bboxes, cur_doc_image_path = [], [], []

            for j in range(len(i_merged_bbox_info)):
                cur_item = i_merged_bbox_info[j]
                cur_doc_lines.append(cur_item['text'])
                # print(cur_item['text'], file_id)
                cur_doc_bboxes.append(self.box_norm(pick_2_points(cur_item['coord']), width=width, height=height))

            total_data['id'] += [file_id, ]
            total_data['lines'] += [cur_doc_lines, ]
            total_data['bboxes'] += [cur_doc_bboxes, ]
            total_data['image_path'] += [file_id, ]

        self.total_data = total_data
        # tokenize text and get bbox/label
        total_input_ids, total_bboxs = [], []
        total_token_len = []

        # 文档 i 的完全体
        for i in range(len(total_data['lines'])):
            cur_doc_id = total_data['id'][i]
            self.doc_id_to_index[cur_doc_id] = i
            cur_doc_input_ids, cur_doc_bboxs = [], []
            now_token_offset = 0
            # bbox
            for j in range(len(total_data['lines'][i])):
                line_encoded = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False,
                                              return_attention_mask=False, return_offsets_mapping=True)
                cur_input_ids = line_encoded['input_ids']
                if len(cur_input_ids) == 0:
                    continue
                cur_offsets_mapping = line_encoded['offset_mapping']

                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                now_token_offset += len(cur_input_ids)

            cur_doc_token_len = now_token_offset

            assert len(cur_doc_input_ids) == len(cur_doc_bboxs), f"{len(cur_doc_input_ids)} == {len(cur_doc_bboxs)} "
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            # 所有token的长度
            total_token_len.append(cur_doc_token_len)

        self.total_inputs = {
            "total_input_ids": total_input_ids,
            "total_bboxs": total_bboxs,
            "total_token_len": total_token_len
        }

        step_size = 255
        max_content_len = 510

        input_ids, bboxs = [], []
        segment_ids, position_ids = [], []

        in_doc_token_offset = []
        doc_ids = []

        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
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

                in_doc_token_offset.append(start)
                doc_ids.append(total_data['id'][i])

                cur_segment_ids = self.get_segment_ids(bboxs[-1])
                cur_position_ids = self.get_position_ids(cur_segment_ids)

                segment_ids.append(cur_segment_ids)
                position_ids.append(cur_position_ids)
                image_path.append(os.path.join(self.args.data_dir, "images", total_data['image_path'][i]))

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

        assert len(input_ids) == len(bboxs) == len(segment_ids) == len(position_ids)
        assert len(segment_ids) == len(image_path)

        res = {
            'input_ids': input_ids,
            'bbox': bboxs,
            'segment_ids': segment_ids,
            'position_ids': position_ids,
            'image_path': image_path,
            'in_doc_token_offset': in_doc_token_offset,
            'doc_ids': doc_ids
        }
        self.data = res
        return res

    def __init__(
            self,
            args,
            ids_order,
            tokenizer,
            image_path,
            max_records=None,
            real_ocr_path=None
    ):
        self.args = args
        self.mode = 'test'
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

        # 组装任务数据 data_file, key 为id, value 就是排好序的bbox_info 应该就OK了
        file_data = {}
        if max_records is not None:
            ids_order = ids_order[:max_records]

        for file_id in ids_order:
            file_data[file_id] = collect_file_task_info(file_id, real_ocr_path)

        self.feature = self.load_data(file_data)
        logging.info(
            f"loading test tasks, get {len(file_data)} examples, get {len(self.feature['input_ids'])} records")

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        # attention_mask = self.feature["attention_mask"][index]
        attention_mask = [1] * len(input_ids)
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        img = pil_loader(self.feature['image_path'][index])
        #for_patches, _ = self.common_transform(img, augmentation=False)
        # patch = self.patch_transform(for_patches)
        patch = self.patch_transform(img)


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
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "images": patch,
            'in_doc_token_offset': in_doc_token_offset,
            'doc_ids': doc_ids
        }
        return res