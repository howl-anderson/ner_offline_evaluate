import collections
import pickle
import sys
from typing import Callable

from tokenizer_tools.tagset.offset.corpus import Corpus

from deliverable_model.serving import SimpleModelInference


def evaluate_offline(model: Callable, corpus: Corpus):
    sample_right = []
    sample_total = []
    sample_decode_failed = []
    span_total = collections.defaultdict(list)
    span_right = collections.defaultdict(list)

    input_for_predict = ["".join(i.text) for i in corpus]

    predict_result = list(model(input_for_predict))

    for gold, predict_result in zip(corpus, predict_result):
        result = predict_result.sequence
        failed = predict_result.is_failed

        sample_total.append(gold)

        if failed:
            sample_decode_failed.append(gold)
        else:
            if gold == result:
                sample_right.append(gold)

        for span in gold.span_set:
            span_total[span.entity].append(span)

            if span in result.span_set:
                span_right[span.entity].append(span)

    return sample_total, sample_right, sample_decode_failed, span_total, span_right


if __name__ == "__main__":
    server = SimpleModelInference(sys.argv[1])
    corpus = Corpus.read_from_file(sys.argv[2])

    sample_total, sample_right, sample_decode_failed, span_total, span_right = evaluate_offline(server.parse, corpus)
    sample_correct_rate = len(sample_right) / len(sample_total)

    span_correct_rate = dict()
    for k, v in span_total.items():
        span_correct_rate[k] = len(span_right.get(k, list())) / len(v)

    span_count = {k: len(v) for k, v in span_total.items()}

    assert len(span_count) == len(span_correct_rate)

    print(sample_correct_rate)
    print(span_correct_rate)
    print(span_count)

    tags = {}
    for key in span_count.keys():
        tags[key] = {
            'number': span_count[key],
            'correct_rate': span_correct_rate[key]
        }

    report_data = {
        'sentence': {'correct_rate': sample_correct_rate, 'number': len(sample_total), 'decode_failed': len(sample_decode_failed)},
        'tags': tags
    }

    with open('result.pkl', 'wb') as fd:
        pickle.dump(report_data, fd)

