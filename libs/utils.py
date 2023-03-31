#!/usr/bin/env python
# -*- coding: utf-8 -*-  
"""
File descriptions in one line

more informations if needed
"""

import os
import random
import string
import datetime
from contextlib import contextmanager
import time
import logging
import json
import re


@contextmanager
def utils_timer(title, print_it=False):
    t0 = time.time()
    begin_str =  "[utils_timer][{}] - begin@[{}]".format(title, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)))
    logging.info(begin_str)
    if print_it:
        print(begin_str)
    yield
    t1 = time.time()
    end_str =  "[utils_timer][{}] - end@[{}] last for [{:.2f}]s".format(title, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)), t1-t0)
    logging.info(end_str)
    if print_it:
        print(end_str)

@contextmanager
def utils_timer2(title, print_it=False):
    t0 = time.time()
    logging.info(begin_str)
    if print_it:
        print(begin_str)
    yield
    t1 = time.time()
    end_str =  "[utils_timer][{}] - end@[{}] last for [{:.2f}]s".format(title, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)), t1-t0)
    logging.info(end_str)
    if print_it:
        print(end_str)

def random_str(length):
    """random but not UNIQ !"""
    return ''.join(random.choice(string.lowercase) for i in range(length))

def make_sure_dir_there(dir_path):
    "check, and create dir if nonexist"
    ret_val = True
    if not os.path.exists(dir_path):
        try:
            # create directory (recursively)
            os.makedirs(dir_path)
        except OSError:
            ret_val = False
    return ret_val

def now_date_str(format_str="%Y%m%d%H%M%S"):
    return datetime.datetime.strftime(datetime.datetime.now(), format_str)

def load_classes_from_txt(class_txt_path):
    classes = []
    class_dict = {}
    with open(class_txt_path) as f:
        for l in f:
            c = l.strip()
            classes.append(c)
            class_dict[c] = len(classes) - 1
    return classes, class_dict


def load_classes_dataset(class_excel_path, col='name'):
    """
    loading AllNLI dataset.
    """
    df = pd.read_excel(class_excel_path)
    classes = sorted(list(set(df[col])))
    classes_dict = dict([(v, k) for k, v in list(enumerate(classes))])
    return classes, classes_dict


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        recs = json.load(f)
    logging.info("loaded {} recs from {}".format(len(recs), file_path))
    return recs


def load_alias_file(file_path):
    d = {}
    re_sp = re.compile(r'\s+')
    with open(file_path) as f:
        for l in f:
            l = l.strip()
            if len(l) == 0 or l[0] == '#':
                continue
            l_sp = re_sp.split(l, 1)
            if len(l_sp) == 2:
                d[l_sp[0].upper()] = l_sp[1]
    return d


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def add_cn(s, d):
    """
        进来之前，s已经做完了字符串的的全角转半角，字符串的大小写转换;
    """
    offset = 0
    for span in re.finditer(r"((\w|'')+?[A-Z]+)", s):
        if span[0] in d:
            s = s[:span.end()+offset] + d[span[0]] + s[span.end()+offset:]
            offset += len(d[span[0]])
    return s


def cdn_pre_process_str(text, d={}):
    text = text.upper()
    text = strQ2B(text)
    return add_cn(text, d)


def cdn_pre_process_recs(recs, d={}):
    for rec in recs:
        rec['text'] = cdn_pre_process_str(rec['text'], d)
    return recs

if __name__ == '__main__':
    pass

