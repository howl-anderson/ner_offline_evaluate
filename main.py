import collections
import pickle
import sys

from tokenizer_tools.conllz.iterator_reader import conllx_iterator_reader
from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder

from seq2annotation.server.tensorflow_inference import Inference
from tokenizer_tools.tagset.offset.sequence import Sequence


def evaluate_offline(model, corpus):
    sample_right = []
    sample_total = []
    span_total = collections.defaultdict(list)
    span_right = collections.defaultdict(list)

    for sample in corpus:
        sample_total.append(sample)

        user_input = sample.text
        gold = sample

        try:
            _, result, _, _ = model(user_input, raise_exception=True)
        except ValueError:
            result = Sequence(user_input)

        if gold == result:
            sample_right.append(gold)

        for span in gold.span_set:
            span_total[span.entity].append(span)

            if span in result.span_set:
                span_right[span.entity].append(span)

    return sample_total, sample_right, span_total, span_right


def read_corpus(data_file):
    corpus = []
    encoder = BILUOSequenceEncoderDecoder()
    for conll_data in conllx_iterator_reader([data_file]):
        char_list = conll_data.word_lines
        tag_list = conll_data.attribute_lines[0]
        # for line in conll_data:
        #     char_list.append(line[0])
        #     tag_list.append(line[1])

        text = ''.join(char_list)

        try:
            seq = encoder.to_offset(tag_list, text)
        except ValueError:
            # print(conll_data)
            # print(text)
            # print(tag_list)
            # print(conll_data.id)
            raise

        corpus.append(seq)

    return corpus


if __name__ == "__main__":
    server = Inference(sys.argv[1])
    corpus = read_corpus(sys.argv[2])

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

