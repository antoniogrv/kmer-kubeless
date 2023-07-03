from typing import Optional
from typing import Final
from typing import Tuple
from typing import Dict
from typing import List
from typing import Any

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from tabulate import tabulate
from logging import log
from glob import glob
import numpy as np
import time
import copy
import os

N_EPOCHS_FOR_CHECKPOINT: Final = 5


class MyModel(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            model_dir: str,
            model_name: str,
            hyperparameter: Dict[str, any],
            weights: Optional[torch.Tensor]
    ):
        super().__init__()
        self.__model_dir = model_dir
        self.__model_name = model_name
        self.__hyperparameter = hyperparameter
        self.__weights = weights

    @abstractmethod
    def load_data(self, batch, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def step(self, inputs: Dict[str, any]) -> any:
        pass

    @abstractmethod
    def compute_loss(self, output, target: torch.Tensor) -> torch.Tensor:
        pass

    def check_checkpoint(self) -> [Tuple[str, int]]:
        # find all model checkpoints path inside model directory
        models_checkpoint_paths: List[str] = glob(
            os.path.join(self.__model_dir, f'{self.__model_name}_*.h5')
        )
        # if there is not a checkpoint return false
        if len(models_checkpoint_paths) == 0:
            return '', 0
        # otherwise, return the last model checkpoint and number of epochs when the last model is saved
        else:
            last_model_path: str = models_checkpoint_paths[-1]
            last_epoch_done: int = int(last_model_path[last_model_path.rindex('_') + 1:-3])
            return last_model_path, last_epoch_done

    def train_model(
            self,
            train_loader: DataLoader,
            optimizer,
            device: torch.device,
            epochs: int = 10,
            evaluation: bool = False,
            val_loader: Optional[DataLoader] = None,
            patience: int = 10,
            logger: Optional[log] = None
    ) -> None:
        # print the header of the result table
        if logger is not None:
            logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | "
                        f"{'Val Loss':^10} | {'Val Acc':^9} | {'Patience':^8} | {'Elapsed':^9}")
            logger.info("-" * 80)

        # init trigger_times and last_loss for early stopping
        last_loss: float = np.inf
        trigger_times: int = 0
        best_model: Optional[MyModel] = None

        # check if there is a checkpoint saved
        __model_checkpoint_path, __last_epoch_done = self.check_checkpoint()
        if __last_epoch_done > 0:
            # load state dict
            __model_checkpoint: MyModel = torch.load(
                __model_checkpoint_path
            )
            self.load_state_dict(__model_checkpoint.state_dict())
            # update n_epochs
            epochs = epochs - __last_epoch_done

        for epoch_i in range(epochs):
            # measure the elapsed time of each epoch
            t0_epoch: float = time.time()
            t0_batch: float = time.time()
            # reset tracking variables at the beginning of each epoch
            total_loss: float = 0
            batch_loss: float = 0
            batch_counts: int = 0

            # put the model into the training mode [IT'S JUST A FLAG]
            self.train()

            # for each batch of training data...
            for step, batch in enumerate(train_loader):
                batch_counts += 1
                # zero out any previously calculated gradients
                optimizer.zero_grad()

                # load batch to GPU
                inputs, target = self.load_data(batch, device)

                # compute loss and accumulate the loss values
                outputs = self.step(inputs)
                loss = self.compute_loss(target, outputs)
                batch_loss += loss.item()
                total_loss += loss.item()

                # perform a backward pass to calculate gradients
                loss.backward()
                # prevent the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                # update parameters and the learning rate
                optimizer.step()

                # print the loss values and time elapsed for every k batches
                if (step % 50 == 0 and step != 0) or (step == len(train_loader) - 1):
                    # calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # print training results
                    if logger is not None:
                        logger.info(
                            f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | "
                            f"{'-':^8} | {time_elapsed:^9.2f}")

                    # reset batch tracking variables
                    batch_loss: float = 0
                    batch_counts: int = 0
                    t0_batch: float = time.time()

            # calculate the average loss over the entire training data
            avg_train_loss: float = total_loss / len(train_loader)

            if logger is not None:
                logger.info("-" * 80)

            if evaluation:
                # after the completion of each training epoch,
                # measure the model's performance on our validation set.
                val_loss, val_accuracy = self.evaluate_model(val_loader, device)
                # print performance over the entire training data
                time_elapsed: float = time.time() - t0_epoch
                # early stopping
                if val_loss > last_loss:
                    trigger_times += 1
                else:
                    last_loss: float = val_loss
                    best_model = copy.deepcopy(self)
                    trigger_times: int = 0
                if logger is not None:
                    logger.info(
                        f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | "
                        f"{val_accuracy:^9.2f} | {patience - trigger_times:^8} | {time_elapsed:^9.2f}")
                    logger.info("-" * 80)
                if trigger_times >= patience:
                    break

                # save model each 5 epochs
                if epoch_i % N_EPOCHS_FOR_CHECKPOINT == 0 and epoch_i != 0:
                    torch.save(self if best_model is None else best_model,
                               os.path.join(self.__model_dir, f'{self.__model_name}_{epoch_i}.h5'))

        # save final model
        torch.save(self if best_model is None else best_model, os.path.join(
            self.__model_dir,
            f'{self.__model_name}.h5'
        ))

        # delete checkpoints
        for model_checkpoint in glob(os.path.join(self.__model_dir, f'{self.__model_name}_*.h5')):
            os.remove(model_checkpoint)

        if logger is not None:
            logger.info("\nTraining complete!")

    @torch.no_grad()
    def evaluate_model(
            self,
            val_loader: DataLoader,
            device: torch.device
    ) -> (float, float):
        # put the model into the evaluation mode. The dropout layers are disabled during the test time.
        self.eval()
        # tracking variables
        val_loss: List[float] = []
        all_outputs = []
        y_true = []

        # for each batch in our validation set...
        for batch in val_loader:
            # load batch to GPU
            inputs, target = self.load_data(batch, device)
            # compute logits
            outputs = self.step(inputs)
            # compute loss
            loss = self.compute_loss(target, outputs)
            val_loss.append(loss.item())

            # save results
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            y_true.append(target)
            all_outputs.append(outputs)

        # Concatenate logits from each batch
        all_outputs = torch.cat(all_outputs, dim=0)
        y_true = torch.cat(y_true, dim=0)

        # Apply softmax to calculate probabilities
        probs: np.ndarray = F.softmax(all_outputs, dim=1).cpu().numpy()
        y_pred: np.ndarray = np.argmax(probs, axis=1)
        y_true = y_true.cpu().numpy()

        # compute the average accuracy and loss over the validation set.
        val_loss: float = float(np.mean(val_loss))
        val_accuracy: float = accuracy_score(y_true, y_pred)

        return val_loss, val_accuracy

    @torch.no_grad()
    def predict(
            self,
            test_loader: DataLoader,
            device: torch.device
    ) -> (np.ndarray, np.ndarray):

        # put the model into the evaluation mode. The dropout layers are disabled during the test time.
        self.eval()
        # init outputs
        all_outputs = []
        y_true = []

        # for each batch in our validation set...
        for batch in test_loader:
            #  load batch to GPU
            inputs, target = self.load_data(batch, device)
            # compute logits
            outputs = self.step(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            y_true.append(target)
            all_outputs.append(outputs)

        # Concatenate logits from each batch
        all_outputs = torch.cat(all_outputs, dim=0)
        y_true = torch.cat(y_true, dim=0)

        # Apply softmax to calculate probabilities
        y_probs: np.ndarray = F.softmax(all_outputs, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        return y_true, y_probs

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def hyperparameter(self) -> Dict[str, any]:
        return self.__hyperparameter

    @abstractmethod
    def get_n_classes(self) -> int:
        pass

    def print_hyperparameter(self) -> str:
        table: List[List[str, Any]] = [[parameter, value] for parameter, value in self.__hyperparameter.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['hyperparameter', 'value'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'
