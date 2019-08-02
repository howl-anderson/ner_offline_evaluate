import collections
import os
import pickle

from pdf_reports import pug_to_html, write_report


with open('result.pkl', 'rb') as fd:
    data = pickle.load(fd)

# reorder tags data
tags_data = collections.OrderedDict(sorted(data['tags'].items(), key=lambda x: x[1]['number'], reverse=True))

html = pug_to_html(
    "template.pug",
    title='NER 模型性能报告',
    sentence=data['sentence'],
    tags=tags_data
)

# Remove old file
os.remove('report.pdf')

write_report(html, "report.pdf")
