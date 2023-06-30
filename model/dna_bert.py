from typing import Optional
from typing import Tuple
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers import (
    BertModel,
    BertConfig
)

from model import Model


class DNABert(Model):
    def __init__(
            self,
            model_name: str,
            model_path: str,
            hyperparameter: Dict[str, any],
            weights: Optional[torch.Tensor] = None
    ):
        # init configuration of model
        self.__config = BertConfig(
            finetuning_task='dnaprom',
            hidden_act='gelu',
            hidden_dropout_prob=hyperparameter['dropout'],
            hidden_size=hyperparameter['hidden_size'],
            vocab_size=hyperparameter['vocab_size'],
            model_type='bert',
            num_attention_heads=hyperparameter['n_attention_heads'],
            num_beams=hyperparameter['n_beams'],
            num_hidden_layers=hyperparameter['n_hidden_layers'],
            num_labels=hyperparameter['n_classes'],
            num_return_sequences=1,
            rnn=hyperparameter['rnn'],
            rnn_dropout=hyperparameter['dropout'],
            rnn_hidden=hyperparameter['hidden_size'],
            num_rnn_layer=hyperparameter['n_rnn_layers']
        )

        # call super class
        super().__init__(model_name, model_path, hyperparameter, weights)

        # create model from configuration
        self.bert = BertModel(self.__config)
        self.dropout = nn.Dropout(self.__config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.__config.hidden_size, self.__config.num_labels)

        # init loss function
        self.__loss = CrossEntropyLoss(weight=weights)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        # call bert forward
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # extract pooled output
        pooled_output = outputs[1]

        # dropout and linear output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        # (loss), logits, (hidden_states), (attentions)
        return outputs

    def load_data(self, batch, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # prepare input of batch for classifier
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['label'].to(device)

        return {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'token_type_ids': token_type_ids
               }, target

    def step(self, inputs: Dict[str, torch.Tensor]):
        # call self.forward
        return self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

    def compute_loss(self, target: torch.Tensor, *outputs):
        logits: torch.Tensor = outputs[0][0]
        return self.__loss(logits.view(-1, self.hyperparameter['n_classes']), target.view(-1))
