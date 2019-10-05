import collections
import pickle
import sys
from typing import Callable

from tokenizer_tools.tagset.offset.corpus import Corpus

from seq2annotation.server.http import load_predict_fn


def evaluate_offline(model: Callable, corpus: Corpus):
    sample_right = []
    sample_total = []
    span_total = collections.defaultdict(list)
    span_right = collections.defaultdict(list)

    for sample in corpus:
        sample_total.append(sample)

        user_input = "".join(sample.text)
        gold = sample

        raw_input_text, result, tags_seq, failed = model(user_input)

        # TODO(xiaoquan.kong): don't count decode failed result
        if gold == result:
            sample_right.append(gold)

        for span in gold.span_set:
            span_total[span.entity].append(span)

            if span in result.span_set:
                span_right[span.entity].append(span)

    return sample_total, sample_right, span_total, span_right


if __name__ == "__main__":
    server = load_predict_fn(sys.argv[1])
    corpus = Corpus.read_from_file(sys.argv[2])

    sample_total, sample_right, span_total, span_right = evaluate_offline(server.infer, corpus)
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
        'sentence': {'correct_rate': sample_correct_rate, 'number': len(sample_total)},
        'tags': tags
    }

    with open('result.pkl', 'wb') as fd:
        pickle.dump(report_data, fd)

