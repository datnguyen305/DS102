import os
from py_vncorenlp import VnCoreNLP
import json

rdrsegmenter = VnCoreNLP(annotators=["wseg"], save_dir='D:/vncorenlp')

with open('D:/vncorenlp/extracting.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for key, value in data.items():
    for index in range(len(value)):
        data[key][index]['text'] = rdrsegmenter.word_segment(value[index]['text'])

with open('D:/vncorenlp/extracting_segmented.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
