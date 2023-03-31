"""
一些相对常用的数据集函数
        by zhangz@20230330
"""


import os
import json
import itertools
from copy import deepcopy
from collections import defaultdict
from PIL import Image



def box_norm(box, width, height, norm_to=1000):
    def clip(min_num, num, max_num):
        return min(max(num, min_num), max_num)

    x0, y0, x1, y1 = box
    x0 = clip(0, int((x0 / width) * norm_to), norm_to)
    y0 = clip(0, int((y0 / height) * norm_to), norm_to)
    x1 = clip(0, int((x1 / width) * norm_to), norm_to)
    y1 = clip(0, int((y1 / height) * norm_to), norm_to)
    # 有些图片旋转的比较厉害 就没有办法咯
    if x1 < x0:
        x1 = x0 + 1
    if y1 < y0:
        y1 = y0 + 1
    assert x1 >= x0 and x0 >= 0, f"{x1}, {x0}"
    assert y1 >= y0 and y0 >= 0, f"{y1}, {y0}"
    return [x0, y0, x1, y1]


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
        简单的位置修正, 固定到合理的矩形坐标上, 副作用是会改动coord原来的数值
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


def c_offset_to_t_offset(offsets_mapping, x):
    """
        char offset -> token offset
        todo: 二分可提速
    :param offsets_mapping:
    :param x:
    :return:
    """
    for i, i_c_offset in enumerate(offsets_mapping):
        if x >= i_c_offset[0] and x < i_c_offset[1]:
            return i
    raise ValueError(f"x should < len(offset_mappings), but {x} >= {len(offsets_mapping)}")

