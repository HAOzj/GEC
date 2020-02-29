# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on FEB 21, 2020

@author: woshihaozhaojun@sina.com
"""
import json
import os
from optparse import OptionParser
from conf_loader import (
    INPUT_TRAIN_FILE, OUTPUT_TRAIN_FILE, NMB_LINES
)


def main_ancien(input_file, output_file, nmb_lines):
    """把fce原始的.json文件的每行转化为{"src": str, "trg": str}的格式

    Args:
        input_file, output_file (path) :- 输入和输出文件的路径
        nmb_lines(int) :- 转化的最大行数
    """
    corpora = []

    with open(input_file, "r") as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            corpora.append(json.loads(line))
            if i >= nmb_lines - 1:
                break

    with open(output_file, "w+") as fp:
        for i, line in enumerate(corpora):
            text = line["text"]
            edits = line["edits"][0][1]
            correction = text
            offset = 0
            for edit in edits:
                [start, end, corr] = edit[: 3]
                start += offset
                end += offset
                if corr:
                    correction = correction[:start] + corr + correction[end:]
                    offset += start - end + len(corr)
                else:
                    correction = correction[:start] + correction[end:]
                    offset += start - end
            content = json.dumps({"src": text, "trg": correction})
            if i < len(corpora) - 1:
                content += "\n"
            fp.write(content)


def main(input_file, output_file):
    """把fce原始的.m2文件的每行转化为{"src": str, "trg": str}的格式

    Args:
        input_file, output_file (path) :- 输入和输出文件的路径
    """
    corpora = []

    with open(input_file, "r") as fp:
        lines = fp.readlines()
        tmp = dict()
        for i, line in enumerate(lines):
            line = line.rstrip("")
            line = line.rstrip("\n")
            if len(line) == 0:
                tmp["edits"] = edits
                corpora.append(tmp)
            elif line[0] == "S":
                tmp = dict()
                tmp["text"] = line[2:]
                edits = []
            elif line[0] == "A":
                edit_list = line[2:].split("|||")
                if edit_list[1] == "noop":
                    pass
                else:
                    edits.append((edit_list[0], edit_list[2]))

    print("一共有{}个句子".format(len(corpora)))

    with open(output_file, "w+") as fp:
        for i, line in enumerate(corpora):
            text = line["text"]
            edits = line["edits"]
            words = text.split(" ")
            edits.reverse()
            if len(edits) == 0:
                correction = text
            else:
                for edit in edits:
                    [start, end] = list(map(lambda x: int(x), edit[0].split(" ")))
                    alter = edit[1]
                    if start == end:
                        words.insert(start, alter)
                    else:
                        words[start: end] = [alter]
                correction = " ".join(words)
            content = json.dumps({"src": text, "trg": correction})
            if i < len(corpora) - 1:
                content += "\n"
            fp.write(content)


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-i', '--input_file', dest='input_file', help='待处理的fce语料的文件路径',
                         type="string", default=INPUT_TRAIN_FILE)
    optparser.add_option('-o', '--output_file', dest='output_file', help='输出的.json文件路径',
                         type="string", default=OUTPUT_TRAIN_FILE)
    # optparser.add_option('-n', '--nmb_lines', dest='nmb_lines', help='处理的语料数',
    #                      type="int", default=NMB_LINES)
    dirname = os.path.dirname(os.path.abspath(__file__))
    (options, args) = optparser.parse_args()
    # main_ancien(options.input_file, options.output_file, options.nmb_lines)
    main(options.input_file, options.output_file)

