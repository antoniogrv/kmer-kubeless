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


class ConvClassifier(MyModel):
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

        # init feature_extractor layer
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        # init classifier
        self.classification = nn.Sequential(
            nn.Linear(
                in_features=63360,
                out_features=1024
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=1024,
                out_features=1 if self.hyperparameter['n_classes'] == 2 else self.hyperparameter['n_classes']
            ),
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
            outputs.append(self.gene_classifier(
                input_ids=matrix_input_ids[idx],
                attention_mask=matrix_attention_mask[idx],
                token_type_ids=matrix_token_type_ids[idx]
            )[0])
        # prepare inputs for fusion classifier
        inputs: torch.Tensor = torch.stack(outputs)  # (batch_size, n_sentences, n_genes)
        batch_size, len_sentence, n_genes = inputs.shape
        inputs = inputs.view(batch_size, len_sentence * n_genes).unsqueeze(1)

        # call future extraction layer
        # let m = len_sentence * n_genes
        # batch_size x 1 x m
        # batch_size x out_channels x (m - kernel_size)
        # batch_size x out_channels / 2 x (m - kernel_size) / 2
        outputs = self.feature_extractor(inputs)

        # use classification layer
        outputs = outputs.view(batch_size, -1)
        outputs = self.classification(outputs)
        # use sigmoid if it binary classification
        if self.get_n_classes() == 2:
            outputs = F.sigmoid(outputs)

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
