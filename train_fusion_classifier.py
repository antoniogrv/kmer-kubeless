from typing import Optional
from typing import Final
from typing import Dict

from dotenv import load_dotenv
import logging
import torch
import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import FusionDataset
from torch.utils.data import DataLoader

from model import MyModel
from model import FCFusionClassifier

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_gene_classifier import train_gene_classifier

from utils import SEPARATOR
from utils import define_fusion_classifier_inputs
from utils import create_test_id
from utils import init_test
from utils import setup_logger
from utils import evaluate_weights
from utils import log_results
from utils import save_result
from utils import close_loggers


def train_fusion_classifier(
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        n_fusion: int,
        classification_type: str,
        gc_model_selected: str,
        gc_hyperparameters: Dict[str, any],
        gc_batch_size: int,
        gc_re_train: bool,
        model_selected: str,
        hyperparameters: Dict[str, any],
        batch_size: int,
        freeze: bool,
        re_train: bool,
        grid_search: bool,
):
    # execute train_gene_classifier
    gc_model_path: str = train_gene_classifier(
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        model_selected=gc_model_selected,
        hyperparameters=gc_hyperparameters,
        batch_size=gc_batch_size,
        re_train=gc_re_train,
        grid_search=True
    )
    # get value from .env
    root_dir: Final = os.getenv('ROOT_LOCAL_DIR')
    # init tokenizer
    tokenizer = Optional[MyDNATokenizer]
    if tokenizer_selected == 'dna_bert':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=False
        )
    elif tokenizer_selected == 'dna_bert_n':
        tokenizer = DNABertTokenizer(
            root_dir=root_dir,
            len_kmer=len_kmer,
            add_n=True
        )
    # generate test id
    test_id: str = create_test_id(
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer,
        gc_hyperparameters=gc_hyperparameters,
        fc_hyperparameters=hyperparameters
    )
    # create dataset configuration
    dataset_conf: Dict[str, any] = FusionDataset.create_conf(
        genes_panel_path=os.getenv('GENES_PANEL_LOCAL_PATH'),
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer,
        n_fusion=n_fusion,
        classification_type=classification_type
    )
    # get global variables
    task: str = os.getenv('FUSION_CLASSIFIER_TASK')
    result_dir: str = os.path.join(os.getcwd(), os.getenv('RESULTS_LOCAL_DIR'))
    model_name: str = os.getenv('MODEL_NAME')
    # init test
    parent_dir, test_dir, log_dir, model_dir, model_path = init_test(
        result_dir=result_dir,
        task=os.path.join(task, classification_type, f'{"freeze" if freeze else "not_freeze"}'),
        model_selected=model_selected,
        test_id=test_id,
        model_name=model_name,
        re_train=re_train
    )

    # if the model has not yet been trained
    if not os.path.exists(model_path):
        # init loggers
        logger: logging.Logger = setup_logger(
            'logger',
            os.path.join(log_dir, 'logger.log')
        )
        train_logger: logging.Logger = setup_logger(
            'train',
            os.path.join(log_dir, 'train.log')
        )

        # load train and validation dataset
        train_dataset = FusionDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='train'
        )
        val_dataset = FusionDataset(
            root_dir=root_dir,
            conf=dataset_conf,
            dataset_type='val'
        )

        # log information
        logger.info(f'Read len: {len_read}')
        logger.info(f'Kmers len: {len_kmer}')
        logger.info(f'Words inside a sentence: {n_words}')
        logger.info(f'Tokenizer used: {tokenizer_selected}')
        logger.info(f'No. fusions generated for each gene: {n_fusion}')
        logger.info(SEPARATOR)
        logger.info(f'Number of train sentences: {len(train_dataset)}')
        logger.info(f'Number of val sentences: {len(val_dataset)}')
        logger.info(f'Number of class: {train_dataset.classes()}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(SEPARATOR)
        logger.info(f'Number of train sentences: {len(train_dataset)}')
        logger.info(f'Number of val sentences: {len(val_dataset)}')
        logger.info(f'Number of class: {train_dataset.classes()}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(SEPARATOR)
        # print dataset status
        logger.info('No. records train set')
        logger.info(train_dataset.print_dataset_status())
        logger.info('No. records val set')
        logger.info(val_dataset.print_dataset_status())

        # load train and validation dataloader
        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # set device gpu if cuda is available
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # evaluating weights for criterion function
        class_weights: torch.Tensor = evaluate_weights(
            train_dataset,
            binary=(classification_type == 'fusion')
        ).to(device)

        # update hyperparameter
        hyperparameters['gene_classifier'] = gc_model_path
        hyperparameters['freeze'] = freeze
        n_kmers: int = len_read - len_kmer + 1
        hyperparameters['n_sentences'] = n_kmers - n_words + 1
        if classification_type == 'fusion':
            hyperparameters['n_classes'] = 2

        # define model
        model: Optional[MyModel] = None
        if model_selected == 'fc':
            model: MyModel = FCFusionClassifier(
                model_dir=model_dir,
                model_name=model_name,
                hyperparameter=hyperparameters,
                weights=class_weights
            )

        # log model hyper parameters
        logger.info('Gene classifier hyperparameter')
        logger.info(model.gene_classifier.print_hyperparameter())
        logger.info('Fusion classifier hyperparameter')
        logger.info(model.print_hyperparameter())

        # init optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode='abs'
        )

        # put model on device available
        model.to(device)
        # train it
        model.train_model(
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=1000,
            evaluation=True,
            scheduler=scheduler,
            val_loader=val_loader,
            logger=train_logger
        )

        # close loggers
        close_loggers([train_logger, logger])
        del train_logger
        del logger

    # if the model is already trained and the grid search parameter is set to true then stop
    elif grid_search:
        return model_path

    # init loggers
    logger: logging.Logger = setup_logger(
        'logger',
        os.path.join(log_dir, 'logger.log')
    )
    result_logger: logging.Logger = setup_logger(
        'result',
        os.path.join(test_dir, 'result.log')
    )

    # load test dataset
    test_dataset = FusionDataset(
        root_dir=root_dir,
        conf=dataset_conf,
        dataset_type='test'
    )

    # log test dataset status
    logger.info('No. records test set')
    logger.info(test_dataset.print_dataset_status())

    # load model
    model: MyModel = torch.load(model_path)
    # set device gpu if cuda is available
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set model on gpu
    model.to(device)

    # create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # test model
    y_true, y_probs = model.predict(
        test_loader=test_loader,
        device=device
    )

    # log results
    y_pred = log_results(
        y_true=y_true,
        y_probs=y_probs,
        target_names=list(test_dataset.get_labels_dict().keys()),
        logger=result_logger,
        test_dir=test_dir
    )

    # close loggers
    close_loggers([logger, result_logger])
    del logger
    del result_logger

    # save result
    save_result(
        result_csv_path=os.path.join(parent_dir, 'results.csv'),
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter=model.hyperparameter,
        y_true=y_true,
        y_pred=y_pred
    )


if __name__ == '__main__':
    # define inputs for this script
    __args, __gc_hyperparameters, __hyperparameters = define_fusion_classifier_inputs()

    # load dotenv file
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

    # execute train_gene_classifier method
    train_fusion_classifier(
        **__args,
        gc_hyperparameters=__gc_hyperparameters,
        hyperparameters=__hyperparameters
    )
