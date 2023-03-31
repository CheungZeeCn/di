import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic, pil_loader
from .dataset_utils import *

XFund_label2ids_gp = {
    "HEADER": 0,
    "QUESTION": 1,
    "ANSWER": 2
}


class xfund_dataset_gp(Dataset):
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
                j = i + 1
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

    def load_data(
            self,
            data_file,
            max_records=None
    ):
        # re-org data format
        # 这里的lines 其实就是里面的bbox的内容
        total_data = {"id": [], "lines": [], "bboxes": [], "ner_tags": [], "image_path": []}
        num_records = len(data_file['documents'])
        if max_records is not None:
            num_records = min(max_records, len(data_file['documents']))

        for i in range(num_records):
            width, height = data_file['documents'][i]['img']['width'], data_file['documents'][i]['img'][
                'height']

            cur_doc_lines, cur_doc_bboxes, cur_doc_ner_tags, cur_doc_image_path = [], [], [], []
            for j in range(len(data_file['documents'][i]['document'])):
                cur_item = data_file['documents'][i]['document'][j]
                cur_doc_lines.append(cur_item['text'])
                cur_doc_bboxes.append(box_norm(cur_item['box'], width=width, height=height))
                cur_doc_ner_tags.append(cur_item['label'])
            total_data['id'] += [len(total_data['id'])]
            total_data['lines'] += [cur_doc_lines]
            total_data['bboxes'] += [cur_doc_bboxes]
            total_data['ner_tags'] += [cur_doc_ner_tags]
            total_data['image_path'] += [data_file['documents'][i]['img']['fname']]

        # tokenize text and get bbox/label
        total_input_ids, total_bboxs, total_label_ids = [], [], []
        # 文档
        for i in range(len(total_data['lines'])):
            cur_doc_input_ids, cur_doc_bboxs, cur_doc_labels = [], [], []
            # bbox
            for j in range(len(total_data['lines'][i])):
                cur_input_ids = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False,
                                               return_attention_mask=False)['input_ids']
                if len(cur_input_ids) == 0: continue
                # 转大写
                cur_label = total_data['ner_tags'][i][j].upper()
                # 这里的数据格式里面 同一个bbox 只能有一个 label;
                if cur_label == 'OTHER':
                    # ignore o label
                    cur_labels = [-100] * len(cur_input_ids)
                    # cur_labels = ["O"] * len(cur_input_ids)
                    # for k in range(len(cur_labels)):
                    #     cur_labels[k] = self.label2ids[cur_labels[k]]

                else:
                    # 为了识别出边界
                    cur_labels = [self.label2ids[cur_label] * 2] * len(cur_input_ids)
                    # 左bbox边界 需要配套改 collect gp label 函数
                    cur_labels[0] = cur_labels[0] + 1
                    # cur_labels[0] = self.label2ids['B-' + cur_labels[0]]
                    # for k in range(1, len(cur_labels)):
                    #     cur_labels[k] = self.label2ids['I-' + cur_labels[k]]
                assert len(cur_input_ids) == len([total_data['bboxes'][i][j]] * len(cur_input_ids)) == len(cur_labels)
                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                cur_doc_labels += cur_labels
            assert len(cur_doc_input_ids) == len(cur_doc_bboxs) == len(cur_doc_labels)
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            total_label_ids.append(cur_doc_labels)
        assert len(total_input_ids) == len(total_bboxs) == len(total_label_ids)

        # split text to several slices because of over-length
        input_ids, bboxs, labels = [], [], []
        segment_ids, position_ids = [], []
        gp_labels = []
        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
            while start < len(total_input_ids[i]):
                # DONE: 这部分的截断可以更灵活一些，取box的边界再截断
                end = min(start + 510, len(total_input_ids[i]))
                # 开始判断 end 是否合适, 关注是否文档边界， 单个bbox是否超过510
                if end == len(total_input_ids[i]):
                    # 到文档边界了， 所以不用处理
                    pass
                else:
                    while total_bboxs[i][end] == total_bboxs[i][end - 1] and end != 0:
                        end -= 1
                    # 这时候 end 可能为bbox边界 或者 start(box超长的时候)
                    if end <= start:
                        # box 超长只能截断
                        end = start + 510

                input_ids.append(
                    [self.tokenizer.cls_token_id] + total_input_ids[i][start: end] + [self.tokenizer.sep_token_id])
                bboxs.append([[0, 0, 0, 0]] + total_bboxs[i][start: end] + [[1000, 1000, 1000, 1000]])
                cur_labels = [-100] + total_label_ids[i][start: end] + [-100]
                labels.append(cur_labels)
                cur_segment_ids = self.get_segment_ids(bboxs[-1])
                cur_position_ids = self.get_position_ids(cur_segment_ids)

                cur_gp_labels = self.collect_gp_labels(cur_labels)

                segment_ids.append(cur_segment_ids)
                position_ids.append(cur_position_ids)
                image_path.append(os.path.join(self.args.data_dir, "images", total_data['image_path'][i]))
                gp_labels.append(cur_gp_labels)

                start = end
                cur_iter += 1

        assert len(input_ids) == len(bboxs) == len(labels) == len(segment_ids) == len(position_ids) == len(gp_labels)
        assert len(segment_ids) == len(image_path)

        # 1. 长文本截断分片，无overlap(这部分在 NER 应该影响不大，但是关系抽取可能会有一定影响);
        # 2. 这里labels仅仅做了id的转化，后续还需要进一步处理, 考虑到最终位移的情况 #
        # 3. gp_labels 列表(batch_size)的列表(single record)的列表(i,j,label), [ [i, j, lable_id], [i,j,label_id]]
        res = {
            'input_ids': input_ids,
            'bbox': bboxs,
            'ori_labels': labels,
            'labels': gp_labels,
            'segment_ids': segment_ids,
            'position_ids': position_ids,
            'image_path': image_path,
        }
        return res

    def __init__(
            self,
            args,
            tokenizer,
            mode,
            max_records=None
    ):
        self.args = args
        self.mode = mode
        self.cur_la = args.language
        self.tokenizer = tokenizer
        self.label2ids = XFund_label2ids_gp

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

        data_file = json.load(
            open(os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val')),
                 'r'))

        self.feature = self.load_data(data_file, max_records)

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        # attention_mask = self.feature["attention_mask"][index]
        attention_mask = [1] * len(input_ids)
        labels = self.feature["labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        img = pil_loader(self.feature['image_path'][index])
        for_patches, _ = self.common_transform(img, augmentation=False)
        patch = self.patch_transform(for_patches)

        assert len(input_ids) == len(attention_mask) == len(bbox) == len(segment_ids), \
            ("len(input_ids):{} == len(attention_mask):{} == len(labels):{}"
             " == len(bbox):{} == len(segment_ids):{}").format(
                len(input_ids), len(attention_mask), len(bbox), len(segment_ids))

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "images": patch,
        }
        return res


