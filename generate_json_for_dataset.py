# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on FEB 21, 2020

@author: woshihaozhaojun@sina.com
"""
import json
import os
from optparse import OptionParser
INPUT_FILE = "data/raw/fce/json/fce.train.json"
OUTPUT_FILE = "data/processed/fce_processed_train.json"
N = 100


def main(input_file, output_file, nmb_lines):
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


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-i', '--input_file', dest='input_file', help='待处理的fce语料的文件路径',
                         type="string", default=INPUT_FILE)
    optparser.add_option('-o', '--output_file', dest='output_file', help='输出的.json文件路径',
                         type="string", default=OUTPUT_FILE)
    optparser.add_option('-n', '--nmb_lines', dest='nmb_lines', help='处理的语料数',
                         type="int", default=N)
    dirname = os.path.dirname(os.path.abspath(__file__))
    (options, args) = optparser.parse_args()
    main(options.input_file, options.output_file, options.nmb_lines)
