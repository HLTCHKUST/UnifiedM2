from __future__ import absolute_import, division, print_function, unicode_literals

from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH
from collections import Counter
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
from tqdm import tqdm
import re
import numpy as np

import torch
import os
import pickle
import json
import jsonlines
import copy
import random

from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)

from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import logging

logger = logging.getLogger(__name__)

from utils.const import LABELS, task2idx

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, task=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task = task

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label, task=None, guid=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.task = task
        self.guid = guid

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AllCombinedProcessor():
    def __init__(self, args):
        # webis
        self.webis_pr = WebisProcessor(args)
        # clickbait
        self.clickbait_pr = ClickbaitProcessor(args)
        # basil_detection
        self.basil_detection_pr = BasilBiasDetectionProcessor(args)
        # basil_type
        self.basil_type_pr = BasilBiasTypeProcessor(args)
        # basil_polarity
        self.basil_polarity_pr = BasilPolarityProcessor(args)
        # rumour_veracity_binary
        self.rumour_binary = RumourVeracityBinaryProcessor(arg)

    def get_train_examples(self):
        webis = self.webis_pr.get_train_examples()
        clickbait = self.clickbait_pr.get_train_examples()
        basil_detection = self.basil_detection_pr.get_train_examples()
        basil_type = self.basil_type_pr.get_train_examples()
        basil_polarity = self.basil_polarity_pr.get_train_examples()
        rumour_binary = self.rumour_binary.get_train_examples()

        return webis, clickbait, basil_detection, basil_type, \
        basil_polarity, rumour_binary

    def get_dev_examples(self):
        webis = self.webis_pr.get_dev_examples()
        clickbait = self.clickbait_pr.get_dev_examples()
        basil_detection = self.basil_detection_pr.get_dev_examples()
        basil_type = self.basil_type_pr.get_dev_examples()
        basil_polarity = self.basil_polarity_pr.get_dev_examples()
        rumour_binary = self.rumour_binary.get_dev_examples()

        return webis, clickbait, basil_detection, basil_type, \
        basil_polarity, rumour_binary

    def get_test_examples(self):
        webis = self.webis_pr.get_test_examples()
        clickbait = self.clickbait_pr.get_test_examples()
        basil_detection = self.basil_detection_pr.get_test_examples()
        basil_type = self.basil_type_pr.get_test_examples()
        basil_polarity = self.basil_polarity_pr.get_test_examples()
        rumour_binary = self.rumour_binary.get_test_examples()

        return webis, clickbait, basil_detection, basil_type, \
        basil_polarity, rumour_binary

class LiarProcessor():
    def __init__(self, args):
        self.train_dir = "{}/preprocessed/liar_train.pickle".format(args.data_dir)
        self.dev_dir = "{}/preprocessed/liar_dev.pickle".format(args.data_dir)
        self.test_dir = "{}/preprocessed/liar_test.pickle".format(args.data_dir)

    def get_train_examples(self):
        with open(self.train_dir, 'rb') as handle:
            train = pickle.load(handle)
        # return self._create_examples(train[:500], "train")
        return self._create_examples(train, "train")

    def get_dev_examples(self):
        with open(self.dev_dir, 'rb') as handle:
            dev = pickle.load(handle)
        # return self._create_examples(dev[:50], "dev")
        return self._create_examples(dev, "dev")

    def get_test_examples(self):
        with open(self.test_dir, 'rb') as handle:
            test = pickle.load(handle)
        # return self._create_examples(test[:50], "test")
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ['half-true', 'false', 'mostly-true', 'barely-true', 'true', 'pants-fire']
        # return ['true', 'false', 'mid']

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i #"%s-%s" % (set_type, i)
            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='liar'))
        return examples

class WebisProcessor():
    def __init__(self, args):
        self.idx=0
        data_dir = '{}/preprocessed/webis.pickle'.format(args.data_dir)
        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)

        if args.do_cross_val:
            kf = KFold(n_splits=args.cross_val_k, shuffle=True, random_state=0)
            splits = list(kf.split(data))
            (train_idx, test_idx) = splits[args.cv_split_idx]
            train = np.array(data)[train_idx]
            self.test = np.array(data)[test_idx]
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
        else:
            train, self.dev = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.test = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        # return ['mostly true', 'no factual content', 'mixture of true and false', 'mostly false']
        return ['true', 'false']

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['text']
            label = obj['veracity_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='webis'))
        return examples

class ClickbaitProcessor():
    def __init__(self, args):
        self.idx = 0
        data_dir = '{}/preprocessed/clickbait_sns.pickle'.format(args.data_dir)

        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)
            train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ["no-clickbait", "clickbait"]

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = " ".join(obj['postText'])
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='clickbait'))
        return examples

class BasilBiasDetectionProcessor():
    def __init__(self, args):
        data_dir = '{}/preprocessed/basil/bias_existence_basil.pickle'.format(args.data_dir)

        # with open("{}/preprocessed_ne/basil/bias_type_basil_label_for_analysis.pickle".format(args.data_dir), 'rb') as handle:
        #     type_labels = pickle.load(handle)

        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)
            new_data = data

            # new_data = []
            # for idx, (d, type_obj) in enumerate(zip(data, type_labels)):
            #     d['type'] = type_obj['label']
            #     # d['id'] = idx
            #     # print(idx)
            #     new_data.append(d)

            if args.do_cross_val:
                kf = KFold(n_splits=args.cross_val_k, shuffle=True, random_state=0)
                splits = list(kf.split(new_data))
                (train_idx, test_idx) = splits[args.cv_split_idx]
                train = np.array(new_data)[train_idx]
                self.test = np.array(new_data)[test_idx]
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
            else:
                train, self.test = train_test_split(new_data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)


    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ["no-bias", "contain-bias"]

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i #obj['id'] # "%s-%s" % (set_type, i)
            text_a = obj['sentence']
            # label = obj['label']
            label = obj['type']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='basil_detection'))
        return examples

class BasilBiasTypeProcessor():
    def __init__(self, args):
        self.idx = 0
        data_dir = '{}/preprocessed/basil/bias_type_basil.pickle'.format(args.data_dir)

        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)
            train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ["Lexical", "Informational"]

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['sentence']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='basil_type'))
        return examples

class BasilPolarityProcessor():
    def __init__(self, args):
        self.idx = 0
        data_dir = '{}/preprocessed/basil/polarity_basil.pickle'.format(args.data_dir)

        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)
            train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ["Negative", "Positive"]

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['sentence']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='basil_polarity'))
        return examples

class BasilBiasSpanProcessor():
    def __init__(self, args):
        self.idx = 0
        data_dir = '{}/preprocessed/basil/bias_span_tagging_basil.pickle'.format(args.data_dir)
        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)
            train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return [] # todo

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['sentence']
            label = obj['text'] # bias span text
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='basil_span'))
        return examples

class FeverProcessor():
    def __init__(self, args):
        self.idx = 0
        # path to data including NEI
        train_dir = '{}/fever/train.jsonl'.format(args.data_dir)
        test_dir = '{}/fever/shared_task_dev.jsonl'.format(args.data_dir)

        train, self.test = [], []
        with jsonlines.open(train_dir) as reader:
            for obj in reader:
                train.append(obj)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

        with jsonlines.open(test_dir) as reader:
            for obj in reader:
                self.test.append(obj)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['claim']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fever'))
        return examples

class FeverBinaryProcessor():
    def __init__(self, args):
        self.idx = 0
        train_dir = '{}/fever/train_verifiable.jsonl'.format(args.data_dir)
        test_dir = '{}/fever/shared_task_dev_ne_verifiable.jsonl'.format(args.data_dir)

        train, self.test = [], []
        with jsonlines.open(train_dir) as reader:
            for obj in reader:
                train.append(obj)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

        with jsonlines.open(test_dir) as reader:
            for obj in reader:
                self.test.append(obj)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['SUPPORTS', 'REFUTES']

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
            text_a = obj['claim']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fever_binary'))
        return examples


class RumourDetectionProcessor():
    def __init__(self, args):
        self.idx = 0
        rumour_path = '{}/preprocessed/event_to_thread_rumour_detection_dataset.pickle'.format(args.data_dir)

        # id2tweet.pickle - has the mappin to whole tweet
        with open('{}/preprocessed/id2text.pickle'.format(args.data_dir), 'rb') as handler:
            self.id2data = pickle.load(handler)

        with open(rumour_path, 'rb') as handle:
            data = pickle.load(handle)
            flatten_data = []
            for event in data.keys():
                flatten_data.extend(data[event])

        train, self.test = train_test_split(flatten_data, test_size=0.1, random_state=0, shuffle=True)
        self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['rumour', 'non-rumour']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            rumour_tweet_id = obj['thread_id']
            text_a = self.id2data[rumour_tweet_id]
            label = obj['detection_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='rumour_detection'))
        return examples

class RumourVeracityProcessor():
    def __init__(self, args):
        veracity_path = '{}/preprocessed/thread_annotations.json'.format(args.data_dir)

        # id2tweet.pickle - has the mappin to whole tweet
        with open('{}/preprocessed/id2text.pickle'.format(args.data_dir), 'rb') as handler:
            self.id2data = pickle.load(handler)
        
        with open(veracity_path) as json_file:
            data = json.load(json_file)
            rumour_data = [data[key] for key in data if data[key]['rumour_label']=='rumour']

        train, self.test = train_test_split(rumour_data, test_size=0.1, random_state=0, shuffle=True)
        self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['false', 'true', 'unverified']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            rumour_tweet_id = obj['thread_id']
            text_a = self.id2data[rumour_tweet_id]
            label = obj['veracity_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='rumour_veracity'))
        return examples

class RumourVeracityByEventProcessor():
    def __init__(self, args):
        self.idx = 0

        self.events = [
                "gurlitt",
                "germanwings",
                "ebola",
                "prince",
                "ferguson",
                "ottawashooting",
                "sydneysiege",
                "charliehebdo",
                "putinmissing"
            ]

        # id2tweet.pickle - has the mappin to whole tweet
        with open('{}/preprocessed/id2text.pickle'.format(args.data_dir), 'rb') as handler:
            self.id2data = pickle.load(handler)

        veracity_path = '{}/preprocessed/thread_annotations_by_event.json'.format(args.data_dir)
        with open(veracity_path) as json_file:
            data = json.load(json_file)

        # split the train/dev/test based on the event

        # prepare test idx
        self.test_event_name = args.test_event_name
        self.test = list(data[self.test_event_name].values())

        # prepare train/dev idx
        train_dev_list = []
        for event_name in self.events:
            if event_name != self.test_event_name:
                train_dev_list.extend(list(data[event_name].values()))

        self.train, self.dev = train_test_split(train_dev_list, test_size=0.15, random_state=0, shuffle=True)

        # self.train = [data[idx] for idx in train_idx_list]
        # self.dev = [data[idx] for idx in dev_idx_list]
        # self.test = [data[idx] for idx in test_idx_list]

        
    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['false', 'true']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)
        
            rumour_tweet_id = obj['thread_id']
            text_a = self.id2data[rumour_tweet_id]
            label = obj['veracity_label']

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='rumour_veracity_binary'))
        return examples




class RumourVeracityBinaryProcessor():
    def __init__(self, args):
        self.idx = 0
        veracity_path = '{}/preprocessed/thread_annotations.json'.format(args.data_dir)

        # id2tweet.pickle - has the mappin to whole tweet
        with open('{}/preprocessed/id2text.pickle'.format(args.data_dir), 'rb') as handler:
            self.id2data = pickle.load(handler)

        with open(veracity_path) as json_file:
            data = json.load(json_file)
            rumour_data = [data[key] for key in data if data[key]['rumour_label']=='rumour']

        # filter 'unverified'
        filtered_rumour_data = []
        for d in rumour_data:
            if d['veracity_label'] == 'unverified':
                continue
            else:
                filtered_rumour_data.append(d)

        # print(len(filtered_rumour_data))
        if args.do_cross_val:
            kf = KFold(n_splits=args.cross_val_k, shuffle=True, random_state=0)
            splits = list(kf.split(filtered_rumour_data))
            (train_idx, test_idx) = splits[args.cv_split_idx]
            train = np.array(filtered_rumour_data)[train_idx]
            self.test = np.array(filtered_rumour_data)[test_idx]

            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
        else:
            train, self.test = train_test_split(filtered_rumour_data, test_size=0.1, random_state=0, shuffle=True)
            # print(len(self.test))
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['false', 'true']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            rumour_tweet_id = obj['thread_id']
            text_a = self.id2data[rumour_tweet_id]
            label = obj['veracity_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='rumour_veracity_binary'))
        return examples


class FakeNewsNetPolitifactProcessor():
    def __init__(self, args):
        with open('{}/preprocessed/fnn_politifact.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        fews_shot_using_train_ratio = False
        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

        self.task_name=args.task_name

    def get_train_examples(self):
        if self.task_name == 'fnn_politifact':
            return self._create_article_examples(self.train, "train")
        elif self.task_name == 'fnn_politifact_title':
            return self._create_title_examples(self.train, "train")

    def get_dev_examples(self):
        if self.task_name == 'fnn_politifact':
            return self._create_article_examples(self.dev, "dev")
        elif self.task_name == 'fnn_politifact_title':
            return self._create_title_examples(self.dev, "dev")

    def get_test_examples(self):
        if self.task_name == 'fnn_politifact':
            return self._create_article_examples(self.test, "test")
        elif self.task_name == 'fnn_politifact_title':
            return self._create_title_examples(self.test, "test")


    def get_labels(self):
        return ['fake', 'real']

    def _create_article_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fnn_politifact'))
        return examples

    def _create_title_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['title']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fnn_politifact_title'))
        return examples

class FakeNewsNetBuzzFeedProcessor():
    def __init__(self, args):
        with open('{}/preprocessed/fnn_buzzfeed.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)
        self.task_name = args.task_name

        fews_shot_using_train_ratio = False
        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)


    def get_train_examples(self):
        if self.task_name == 'fnn_buzzfeed':
            return self._create_article_examples(self.train, "train")
        elif self.task_name == 'fnn_buzzfeed_title':
            return self._create_title_examples(self.train, "train")
    def get_dev_examples(self):
        if self.task_name == 'fnn_buzzfeed':
            return self._create_article_examples(self.dev, "dev")
        elif self.task_name == 'fnn_buzzfeed_title':
            return self._create_title_examples(self.dev, "dev")
    def get_test_examples(self):
        if self.task_name == 'fnn_buzzfeed':
            return self._create_article_examples(self.test, "test")
        elif self.task_name == 'fnn_buzzfeed_title':
            return self._create_title_examples(self.test, "test")
            
    def get_labels(self):
        return ['fake', 'real']

    def _create_article_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fnn_buzzfeed'))
        return examples

    def _create_title_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['title']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fnn_buzzfeed_title'))
        return examples

class FakeNewsNetGossipProcessor():
    def __init__(self, args):
        with open('{}/preprocessed/fnn_gossip.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        if not args.do_zeroshot:
            train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
        else:
            self.train, self.dev = None, None
            self.test = data


    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['real', 'fake']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i #"%s-%s" % (set_type, i)

            text_a = obj["title"]
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='fnn_gossip'))
        return examples


class PropagandaProcessor():
    def __init__(self, args):
        with open('{}/preprocessed/propaganda_train.pickle'.format(args.data_dir), 'rb') as handler:
            train = pickle.load(handler)

            fews_shot_using_train_ratio = False

            if fews_shot_using_train_ratio:
                # few shot experiment with train shot RATIO
                chunk_indx = int(len(train) * args.fewshot_train_ratio)
            else:
                # few shot experiment with train shot SIZE
                random.seed(args.seed)
                random.shuffle(train)
                chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)
                    
            self.train = train[:chunk_indx]

            # if chunk_indx > 0:
            #     self.train = train[:chunk_indx]
            # else:
            #     # full data
            #     self.train = train


        with open('{}/preprocessed/propaganda_dev.pickle'.format(args.data_dir), 'rb') as handler:
            self.dev = pickle.load(handler)

        with open('{}/preprocessed/propaganda_test.pickle'.format(args.data_dir), 'rb') as handler:
            self.test = pickle.load(handler)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['has_propaganda', 'no_propaganda']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i #"%s-%s" % (set_type, i)

            text_a = obj["text"]
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='propaganda'))
        return examples


class CovidTwitter_Q1_Processor():
    def __init__(self, args):
        with open('{}/preprocessed/twitter_q1.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        fews_shot_using_train_ratio = False
        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['yes', 'no']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='covid_twitter_q1'))
        return examples


class CovidTwitter_Q2_Processor():
    def __init__(self, args):
        with open('{}/preprocessed/twitter_q2_mapped.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        fews_shot_using_train_ratio = False

        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
      

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['no_false', 'contains_false']


    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='covid_twitter_q2'))
        return examples


class CovidTwitter_Q6_Processor():
    def __init__(self, args):
        with open('{}/preprocessed/twitter_q6_mapped.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        fews_shot_using_train_ratio = False

        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)
         

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['not_harmful', 'harmful']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='covid_twitter_q6'))
        return examples


class CovidTwitter_Q7_Processor():
    def __init__(self, args):
        with open('{}/preprocessed/twitter_q7_mapped.pickle'.format(args.data_dir), 'rb') as handler:
            data = pickle.load(handler)

        fews_shot_using_train_ratio = False

        if fews_shot_using_train_ratio:
            # few shot experiment with train shot RATIO
            self.train, test = train_test_split(data, test_size=1-args.fewshot_train_ratio, random_state=args.seed, shuffle=True)
            self.dev, self.test = train_test_split(test, test_size=0.8, random_state=args.seed, shuffle=True)
        else:
            # few shot experiment with train shot SIZE
            random.seed(args.seed)
            random.shuffle(data)
            chunk_indx = args.fewshot_train # int(len(train) * args.fewshot_train_ratio)

            if chunk_indx != None:
                self.train = data[:chunk_indx]
                self.dev, self.test = train_test_split(data[chunk_indx:], test_size=0.8, random_state=0, shuffle=True)
            else:
                # full data
                train, self.test = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)
                self.train, self.dev = train_test_split(train, test_size=0.15, random_state=0, shuffle=True)

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_dev_examples(self):
        return self._create_examples(self.dev, "dev")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return ['not_attention', 'attention']

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            if i == 0:
                continue
            guid = i # "%s-%s" % (set_type, i)

            text_a = obj['text']
            label = obj['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='covid_twitter_q7'))
        return examples



def get_all_misinfo_dataset(args, tokenizer, phase='train'):
    processor = AllCombinedProcessor(args)

    if phase == 'train':
        all_examples = processor.get_train_examples()
    elif phase == 'dev':
        all_examples = processor.get_dev_examples()
    else:
        all_examples = processor.get_test_examples()
        all_examples = all_examples #[:50]

    combined_all_input_ids, combined_all_attention_mask, combined_all_token_type_ids = [],[],[]
    combined_all_labels, combined_task_idx, combined_all_guids = [],[],[]
    for examples in all_examples:
        features = convert_misinfo_examples_to_features(examples,
                                                tokenizer,
                                                remove_stopwords=args.remove_stopwords,
                                                label_map=LABELS[task],
                                                max_length=args.max_seq_length,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        task_idx = torch.tensor([task2idx[task] for _ in features], dtype=torch.long)
        all_guids = torch.tensor([f.guid for f in features])

        combined_all_input_ids.append(all_input_ids)
        combined_all_attention_mask.append(all_attention_mask)
        combined_all_token_type_ids.append(all_token_type_ids)
        combined_all_labels.append(all_labels)
        combined_task_idx.append(task_idx)
        combined_all_guids.append(all_guids)

    dataset = TensorDataset(combined_all_input_ids, combined_all_attention_mask, combined_all_all_token_type_ids, combined_all_all_labels, combined_all_task_idx, combined_all_all_guids)

    return dataset


def get_misinfo_datset(args, task, tokenizer, phase='train'):
    if task == 'liar':
        processor = LiarProcessor(args)
    elif task == 'webis':
        processor = WebisProcessor(args)
    elif task == 'clickbait':
        processor = ClickbaitProcessor(args)
    elif task == 'basil_detection':
        processor = BasilBiasDetectionProcessor(args)
    elif task == 'basil_type':
        processor = BasilBiasTypeProcessor(args)
    elif task == 'basil_polarity':
        processor = BasilPolarityProcessor(args)
    elif task == 'fever':
        processor = FeverProcessor(args)
    elif task == 'fever_binary':
        processor = FeverBinaryProcessor(args)
    elif task == 'rumour_detection':
        processor = RumourDetectionProcessor(args)
    elif task == 'rumour_veracity':
        processor = RumourVeracityProcessor(args)
    elif task == 'rumour_binary_byEvents':
        processor = RumourVeracityByEventProcessor(args)
    elif task == 'rumour_veracity_binary':
        processor = RumourVeracityBinaryProcessor(args)
    elif task == 'fnn_politifact' or task == 'fnn_politifact_title':
        processor = FakeNewsNetPolitifactProcessor(args)
    elif task == 'fnn_buzzfeed' or task == 'fnn_buzzfeed_title':
        processor = FakeNewsNetBuzzFeedProcessor(args)
    elif task == 'fnn_gossip':
        processor = FakeNewsNetGossipProcessor(args)
    elif task == 'propaganda':
        processor = PropagandaProcessor(args)
    elif task == 'newstrust':
        pass
        # processor = NewsTrustProcessor(args)
    elif task == 'covid_twitter_q1':
        processor = CovidTwitter_Q1_Processor(args)
    elif task == 'covid_twitter_q2':
        processor = CovidTwitter_Q2_Processor(args)
    elif task == 'covid_twitter_q6':
        processor = CovidTwitter_Q6_Processor(args)
    elif task == 'covid_twitter_q7':
        processor = CovidTwitter_Q7_Processor(args)
    else:
        print("wrong task given: {}".format(task))
        exit(1)


    if phase == 'train':
        examples = processor.get_train_examples()
    elif phase == 'dev':
        examples = processor.get_dev_examples()
    else:
        examples = processor.get_test_examples()
        examples = examples #[:50]

    features = convert_misinfo_examples_to_features(examples,
                                            tokenizer,
                                            remove_stopwords=args.remove_stopwords,
                                            label_map=LABELS[task],
                                            max_length=args.max_seq_length,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    task_idx = torch.tensor([task2idx[task] for _ in features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in features])

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, task_idx, all_guids)

    return dataset

def get_processor(task_name):
    if task_name == 'liar': return LiarProcessor
    if task_name == 'webis': return WebisProcessor
    if task_name == 'clickbait': return ClickbaitProcessor
    if task_name == 'basil_detection': return BasilBiasDetectionProcessor
    if task_name == 'basil_type': return BasilBiasTypeProcessor
    if task_name == 'basil_polarity': return BasilPolarityProcessor
    if task_name == 'fever': return FeverProcessor
    if task_name == 'fever_binary': return FeverBinaryProcessor
    if task_name == 'rumour_detection': return RumourDetectionProcessor
    if task_name == 'rumour_veracity': return RumourVeracityProcessor
    if task_name == 'rumour_veracity_binary': return RumourVeracityBinaryProcessor
    if task_name == 'fnn_politifact': return FakeNewsNetPolitifactProcessor
    if task_name == 'fnn_buzzfeed': return FakeNewsNetBuzzFeedProcessor
    if task_name == 'fnn_gossip': return FakeNewsNetGossipProcessor
    if task_name == 'fnn_buzzfeed_title': return FakeNewsNetBuzzFeedProcessor
    if task_name == 'propaganda': return PropagandaProcessor
    if task_name == 'covid_twitter_q1': return CovidTwitter_Q1_Processor
    if task_name == 'fnn_politifact_title': return FakeNewsNetPolitifactProcessor
    if task_name == 'covid_twitter_q2': return CovidTwitter_Q2_Processor
    if task_name == 'covid_twitter_q6': return CovidTwitter_Q6_Processor
    if task_name == 'covid_twitter_q7': return CovidTwitter_Q7_Processor
    if task_name == 'rumour_binary_byEvents': return RumourVeracityByEventProcessor

def convert_misinfo_examples_to_features(examples, tokenizer, remove_stopwords,
                                      max_length=512,
                                      task=None,
                                      label_map=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            clean_text(example.text_a, remove_stopword=remove_stopwords),
            clean_text(example.text_b, remove_stopword=remove_stopwords) if example.text_b is not None else example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = label_map[example.label]
        task = example.task
        guid = example.guid

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label, task=task, guid=guid))

    return features

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def clean_text(text,
               remove_stopword=False,
              remove_nonalphanumeric=False,
              use_number_special_token=False):
    text = text.lower()
    text = re.sub(r"\n", " ", text)

    if remove_nonalphanumeric:
        text = re.sub(r'([^\s\w\'.,!?"%]|_)+', " ", text)

    if use_number_special_token:
        text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)

    if remove_stopword:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords]
        text = " ".join(tokens)

    # Remove URL
    text = re.sub(r"(http)\S+", "", text)
    text = re.sub(r"(www)\S+", "", text)
    text = re.sub(r"(href)\S+", "", text)
    # Remove multiple spaces
    text = re.sub(r"[ \s\t\n]+", " ", text)
    #
    # # remove repetition
    # text = re.sub(r"([!?.]){2,}", r"\1", text)
    # text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

    return text.strip()

