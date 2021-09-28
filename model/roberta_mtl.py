from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from transformers import BertPreTrainedModel, RobertaModel
from .roberta_single import RobertaConfig
import numpy as np

# from utils.model_utils import layer_freeze

task2idx = {
    'liar': 0,
    'webis': 1,
    'clickbait': 2,
    'basil_detection': 3,
    'basil_type': 4,
    'basil_polarity': 5,
    'fever': 6,
    'fever_binary': 7,
    'rumour_detection': 8,
    'rumour_veracity': 9,
    # 'fnn_politifact': 2,
    # 'fnn_buzzfeed': 2,
    # 'fnn_gossip': 2,
    'rumour_veracity_binary': 10,
    'fnn_politifact': 11,
    'fnn_buzzfeed': 12,
    # 'fnn_gossip': 13,
    'fnn_buzzfeed_title': 14,
    'propaganda': 15,
    'covid_twitter_q1': 16,
    'fnn_politifact_title': 17,

    'covid_twitter_q2': 18,
    'covid_twitter_q6': 19,
    'covid_twitter_q7': 20,
    'new_rumour_binary': 21
}

idx2task = {
    "0": 'liar',
    "1": 'webis',
    "2": 'clickbait',
    "3": 'basil_detection',
    "4": 'basil_type',
    "5": 'basil_polarity',
    "6": "fever",
    "7": "fever_binary",
    "8": "rumour_detection",
    "9": "rumour_veracity",
    '10': "rumour_veracity_binary",
    "11": 'fnn_politifact',
    "12": 'fnn_buzzfeed',
    # "13": 'fnn_gossip',
    "14": 'fnn_buzzfeed_title',
    "15": 'propaganda',
    "16": 'covid_twitter_q1',
    "17": 'fnn_politifact_title',
    "18": 'covid_twitter_q2',
    "19": 'covid_twitter_q6',
    "20": 'covid_twitter_q7',

}

"""MTL model using Roberta"""
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
}

class RobertaForMTL(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMTL, self).__init__(config)

        self.num_labels = config.num_labels
        self.tasks = config.finetuning_task

        self.loss_f = config.loss_f

        self.main_task = self.tasks[0]

        self.num_layers_to_freeze = config.num_layers_to_freeze

        self.num_labels_dict = {task: n_label
                        for task, n_label in zip(self.tasks, config.num_labels)}

        self.roberta = RobertaModel(config)
        # if self.num_layers_to_freeze > 0:
        #     layer_freeze(self.roberta, self.num_layers_to_freeze)

        custom_dropout = config.custom_dropout
        roberta_head_dict = {task: RobertaClassificationHead(config.hidden_size, custom_dropout, n_label)
                            for task, n_label in zip(self.tasks, config.num_labels)}

        self.classifiers = nn.ModuleDict(roberta_head_dict)

    def get_loss(self, logits, labels, task):
        num_labels = self.num_labels_dict[task]
        if num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return loss


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, task=None, guids=None):
        task_name = idx2task[str(task[0].item())]

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]

        # choose the correct classifier to pass through
        logits = self.classifiers[task_name](sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.get_loss(logits, labels, task_name)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
