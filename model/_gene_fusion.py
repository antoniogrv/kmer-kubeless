from typing import Optional
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import BCELoss

from model import MyModel


class FusionClassifier(MyModel):
    def __init__(
            self,
            model_dir: str,
            model_name: str,
            hyperparameter: Dict[str, any],
            weights: Optional[torch.Tensor]
    ):
        # call super class
        super().__init__(model_dir, model_name, hyperparameter, weights)

        # init configuration of model
        self.__gene_classifier_path: str = hyperparameter['gene_classifier']
        # load gene classifier
        self.gene_classifier: MyModel = torch.load(
            self.__gene_classifier_path
        )

        # load configuration
        self.__n_sentences = self.hyperparameter['n_sentences']
        self.__n_genes = self.gene_classifier.get_n_classes()

        # projection layer
        self.projection = nn.Linear(
            in_features=self.__n_sentences * self.__n_genes,
            out_features=self.hyperparameter['hidden_size']
        )

        # classification layer
        self.classification = nn.Linear(
            in_features=self.hyperparameter['hidden_size'],
            out_features=1 if self.hyperparameter['n_classes'] == 2 else self.hyperparameter['n_classes']
        )

        # init loss function
        self.loss = BCELoss(weight=weights)

    def forward(
            self,
            matrix_input_ids=None,
            matrix_attention_mask=None,
            matrix_token_type_ids=None
    ):
        # call bert on each sentence
        outputs: List[torch.Tensor] = []
        for idx in range(len(matrix_input_ids)):
            outputs.append(self.gene_classifier(
                input_ids=matrix_input_ids[idx],
                attention_mask=matrix_attention_mask[idx],
                token_type_ids=matrix_token_type_ids[idx]
            )[0])
        # prepare inputs for fusion classifier
        inputs: torch.Tensor = torch.stack(outputs)  # (batch_size, n_sentences, n_genes)
        inputs = torch.flatten(inputs, start_dim=1, end_dim=2)  # (batch_size, n_sentences * n_genes)
        # use classifier layer
        outputs = self.classification(inputs)

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

    def compute_loss(self, output, target: torch.Tensor):
        pass

    def get_n_classes(self) -> int:
        pass
