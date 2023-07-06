from typing import Optional
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F

from model import MyModel
from model import GeneClassifier


class FCFusionClassifier(MyModel):
    def __init__(
            self,
            model_dir: str,
            model_name: str,
            hyperparameter: Dict[str, any],
            weights: Optional[torch.Tensor]
    ):
        # call super class
        super().__init__(
            model_dir,
            model_name,
            {
                'hidden_size': hyperparameter['hidden_size'],
                'n_hidden_layers': hyperparameter['n_hidden_layers'],
                'dropout': hyperparameter['dropout'],
                'gene_classifier': hyperparameter['gene_classifier'],
                'n_sentences': hyperparameter['n_sentences'],
                'n_classes': hyperparameter['n_classes']
            },
            weights
        )

        # init configuration of model
        self.__gene_classifier_path: str = self.hyperparameter['gene_classifier']
        # load gene classifier
        self.gene_classifier: GeneClassifier = torch.load(
            self.__gene_classifier_path
        )

        # freeze all layer of gene_classifier
        if self.hyperparameter['freeze']:
            for param in self.gene_classifier.get_embedding_layer().parameters():
                param.requires_grad = False

        # load configuration
        self.__n_sentences = self.hyperparameter['n_sentences']

        # projection layer
        self.projection = nn.Linear(
            in_features=self.__n_sentences * self.gene_classifier.hyperparameter['hidden_size'],
            out_features=self.hyperparameter['hidden_size']
        )
        # init fusion classifier layer
        __fusion_classifier_layer = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.hyperparameter['hidden_size'],
                    out_features=self.hyperparameter['hidden_size']
                ),
                nn.Dropout(p=self.hyperparameter['dropout'])
            ]
        )
        # create fusion classifier model
        self.fusion_classifier = nn.ModuleList(
            [
                __fusion_classifier_layer for _ in range(self.hyperparameter['n_hidden_layers'])
            ]
        )

        # classification layer
        self.classification = nn.Linear(
            in_features=self.hyperparameter['hidden_size'],
            out_features=1 if self.hyperparameter['n_classes'] == 2 else self.hyperparameter['n_classes']
        )

        # init loss function
        self.loss = BCEWithLogitsLoss(pos_weight=weights)

    def forward(
            self,
            matrix_input_ids=None,
            matrix_attention_mask=None,
            matrix_token_type_ids=None
    ):
        # call bert on each sentence
        outputs = []
        for idx in range(len(matrix_input_ids)):
            outputs.append(
                self.gene_classifier.embedding_step(
                    {
                        'input_ids': matrix_input_ids[idx],
                        'attention_mask': matrix_attention_mask[idx],
                        'token_type_ids': matrix_token_type_ids[idx]
                    }
                )
            )
        # prepare inputs for fusion classifier
        inputs: torch.Tensor = torch.stack(outputs)  # (batch_size, n_sentences, hidden_size)
        inputs = torch.flatten(inputs, start_dim=1, end_dim=2)

        # use projection layer
        outputs = self.projection(inputs)
        # execute all layers of fusion classifier
        for layer_idx in range(len(self.fusion_classifier)):
            outputs = self.fusion_classifier[layer_idx][0](outputs)
            outputs = self.fusion_classifier[layer_idx][1](outputs)
            outputs = F.gelu(outputs)

        # use classification layer
        outputs = self.classification(outputs)

        return outputs

    def load_data(self, batch, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # prepare input of batch for classifier
        matrix_input_ids = batch['matrix_input_ids'].to(device)
        matrix_attention_mask = batch['matrix_attention_mask'].to(device)
        matrix_token_type_ids = batch['matrix_token_type_ids'].to(device)
        target = batch['label'].to(device)

        return {
                   'matrix_input_ids': matrix_input_ids,
                   'matrix_attention_mask': matrix_attention_mask,
                   'matrix_token_type_ids': matrix_token_type_ids
               }, target

    def step(self, inputs: Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]]]]):
        # call self.forward
        return self(
            matrix_input_ids=inputs['matrix_input_ids'],
            matrix_attention_mask=inputs['matrix_attention_mask'],
            matrix_token_type_ids=inputs['matrix_token_type_ids']
        )

    def compute_loss(self, target: torch.Tensor, *outputs):
        return self.loss(input=outputs[0], target=target.float())

    def get_n_classes(self) -> int:
        return self.hyperparameter['n_classes']
