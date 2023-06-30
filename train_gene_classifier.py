from typing import Optional
from typing import Final
from typing import Dict

import numpy as np
import argparse
import logging
import torch
import os

from tokenizer import MyDNATokenizer
from tokenizer import DNABertTokenizer

from dataset import TranscriptDataset
from torch.utils.data import DataLoader

from model import Model
from model import DNABert

from torch.optim import AdamW
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from utils import SEPARATOR
from utils import str2bool
from utils import create_test_name
from utils import test_check
from utils import create_folders
from utils import setup_logger
from utils import save_result
from utils import close_loggers

TASK: Final = 'gene_classification'


def define_input_args_model_hyperparameters(
        arg_parser: argparse.ArgumentParser,
        suffix: str = ''
) -> None:
    arg_parser.add_argument(f'-{suffix}model_selected', dest=f'{suffix}model_selected', action='store',
                            type=str, default='dna_bert', help='select the model to be used')
    arg_parser.add_argument(f'-{suffix}hidden_size', dest=f'{suffix}hidden_size', action='store',
                            type=int, default=1024, help='define number of hidden channels')
    arg_parser.add_argument(f'-{suffix}n_hidden_layers', dest=f'{suffix}n_hidden_layers', action='store',
                            type=int, default=7, help='define number of hidden layers')
    arg_parser.add_argument(f'-{suffix}rnn', dest=f'{suffix}rnn', action='store',
                            type=str, default='lstm', help='define type of recurrent layer')
    arg_parser.add_argument(f'-{suffix}n_rnn_layers', dest=f'{suffix}n_rnn_layers', action='store',
                            type=int, default=1, help='define number of recurrent layers')
    arg_parser.add_argument(f'-{suffix}n_attention_heads', dest=f'{suffix}n_attention_heads', action='store',
                            type=int, default=4, help='define number of attention heads')
    arg_parser.add_argument(f'-{suffix}n_beams', dest=f'{suffix}n_beams', action='store',
                            type=int, default=1, help='define number of beams')
    arg_parser.add_argument(f'-{suffix}dropout', dest=f'{suffix}dropout', action='store',
                            type=float, default=0.5, help='define value of dropout probability')


def check_gene_classifier_hyperparameters(
        args_dict: Dict[str, str],
        suffix: str = ''
) -> None:
    # check model selected
    if args_dict[f'{suffix}model_selected'] not in ['dna_bert']:
        raise ValueError('select one of these models: ["dna_bert"]')

    # check tokenizer selected
    if args_dict['tokenizer_selected'] not in ['dna_bert', 'dna_bert_n']:
        raise ValueError('select one of these tokenizers: ["dna_bert", "dna_bert_n"]')

    # check recurrent layer selected
    if args_dict[f'{suffix}rnn'] not in ['lstm', 'gru']:
        raise ValueError('select one of these recurrent layers: ["lstm", "gru"]')


def init_hyperparameters_dict(
        args_dict: Dict[str, str],
        suffix: str = ''
) -> Dict[str, any]:
    return {
        'hidden_size': args_dict[f'{suffix}hidden_size'],
        'dropout': args_dict[f'{suffix}dropout'],
        'n_attention_heads': args_dict[f'{suffix}n_attention_heads'],
        'n_beams': args_dict[f'{suffix}n_beams'],
        'n_hidden_layers': args_dict[f'{suffix}n_hidden_layers'],
        'rnn': args_dict[f'{suffix}rnn'],
        'n_rnn_layers': args_dict[f'{suffix}n_rnn_layers']
    }


def train_gene_classifier(
        len_read: int,
        len_kmer: int,
        n_words: int,
        model_selected: str,
        tokenizer_selected: str,
        batch_size: int,
        hyperparameter: Dict[str, any],
        re_train: bool,
        grid_search: bool
):
    # generate test name
    test_name: str = create_test_name(
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter=hyperparameter
    )
    print(f'Test name: {test_name}')

    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # init tokenizer
    tokenizer = Optional[MyDNATokenizer]
    if tokenizer_selected == 'dna_bert':
        tokenizer = DNABertTokenizer(
            root_dir=os.path.join(os.getcwd(), 'data'),
            len_kmer=len_kmer
        )
    elif tokenizer_selected == 'dna_bert_n':
        tokenizer = DNABertTokenizer(
            root_dir=os.path.join(os.getcwd(), 'data'),
            len_kmer=len_kmer,
            add_n=True
        )

    # create dataset configuration
    dataset_conf: Dict[str, any] = TranscriptDataset.create_conf(
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer=tokenizer
    )

    # check if this configuration is already tested
    if re_train or not test_check(task=TASK, model_name=model_selected, parent_name=test_name):
        print(f'Initialization of the test...')
        # create folders and get path
        log_path, model_path = create_folders(
            task=TASK,
            model_name=model_selected,
            parent_name=test_name
        )

        # init loggers
        logger: logging.Logger = setup_logger(
            'logger',
            os.path.join(log_path, 'logger.log')
        )
        train_logger: logging.Logger = setup_logger(
            'train',
            os.path.join(log_path, 'train.log')
        )

        # load train and validation dataset
        train_dataset = TranscriptDataset(
            root_dir=os.path.join(os.getcwd(), 'data'),
            conf=dataset_conf,
            dataset_type='train'
        )
        val_dataset = TranscriptDataset(
            root_dir=os.path.join(os.getcwd(), 'data'),
            conf=dataset_conf,
            dataset_type='val'
        )

        # log information
        logger.info(f'Read len: {len_read}')
        logger.info(f'Kmers len: {len_kmer}')
        logger.info(f'Words inside a sentence: {n_words}')
        logger.info(f'Tokenizer used: {tokenizer_selected}')
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

        # evaluate weights for criterion function
        y = []
        for idx, label in enumerate(train_dataset.get_dataset_status()):
            y = np.append(y, [idx] * label)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights: torch.Tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # update hyperparameter
        hyperparameter['vocab_size'] = tokenizer.vocab_size
        hyperparameter['n_classes'] = train_dataset.classes()

        # define model
        model: Model = DNABert(
            model_name='model',
            model_path=model_path,
            hyperparameter=hyperparameter,
            weights=class_weights
        )
        # log model hyper parameters
        logger.info('Model hyperparameter')
        logger.info(model.print_hyperparameter())

        # init optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            eps=1e-8,
            betas=(0.9, 0.999)
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
            val_loader=val_loader,
            logger=train_logger
        )

        # close loggers
        close_loggers([train_logger, logger])
        del train_logger
        del logger

    # get path of model and log
    log_path, model_path = create_folders(
        task=TASK,
        model_name=model_selected,
        parent_name=test_name
    )

    # if grid search is True and this model is already evaluated, return
    if grid_search and os.path.exists(os.path.join(log_path, 'result.log')):
        return

    # init loggers
    logger: logging.Logger = setup_logger(
        'logger',
        os.path.join(log_path, 'logger.log')
    )
    result: logging.Logger = setup_logger(
        'result',
        os.path.join(log_path, 'result.log')
    )

    # load test dataset
    test_dataset = TranscriptDataset(
        root_dir=os.path.join(os.getcwd(), 'data'),
        conf=dataset_conf,
        dataset_type='test'
    )
    # log test dataset status
    logger.info('No. records test set')
    logger.info(test_dataset.print_dataset_status())

    # load model
    model: Model = torch.load(os.path.join(model_path, 'model.h5'))
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
    y_true, y_pred = model.predict(
        test_loader=test_loader,
        device=device
    )

    # log classification report
    report: str = classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=1,
        target_names=test_dataset.get_labels_dict().keys()
    )
    result.info(report)

    # close loggers
    close_loggers([logger, result])
    del logger
    del result

    # save result
    save_result(
        result_csv_path=os.path.join(os.getcwd(), 'log', TASK, model_selected, 'results.csv'),
        len_read=len_read,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter=model.hyperparameter,
        y_true=y_true,
        y_pred=y_pred
    )


if __name__ == '__main__':
    # init parser for inputs
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('-len_read', dest='len_read', action='store',
                        type=int, default=150, help='define length of reads')
    parser.add_argument('-len_kmer', dest='len_kmer', action='store',
                        type=int, default=6, help='define length of kmers')
    parser.add_argument('-n_words', dest='n_words', action='store',
                        type=int, default=20, help='number of kmers inside a sentence')
    parser.add_argument('-tokenizer_selected', dest='tokenizer_selected', action='store',
                        type=str, default='dna_bert_n', help='select the tokenizer to be used')
    parser.add_argument('-batch_size', dest='batch_size', action='store',
                        type=int, default=512, help='define batch size')

    # gene classifier parameters
    define_input_args_model_hyperparameters(arg_parser=parser)

    # train parameters
    parser.add_argument('-re_train', dest='re_train', action='store', type=str2bool,
                        default=False, help='set true if you wish to retrain the model despite having already '
                                            'tested with these hyperparameters. Obviously, if the model has been '
                                            'trained on a different dataset you need to set this parameter to true')
    parser.add_argument('-grid_search', dest='grid_search', action='store', type=str2bool,
                        default=False, help='set true if this script is launching from grid_search script')

    # parse arguments
    args = parser.parse_args()

    # check value of model hyperparameters
    check_gene_classifier_hyperparameters(
        args_dict=vars(args)
    )

    # init hyperparameters config
    config: Dict[str, any] = init_hyperparameters_dict(
        args_dict=vars(args)
    )

    # execute main
    train_gene_classifier(
        len_read=args.len_read,
        len_kmer=args.len_kmer,
        n_words=args.n_words,
        model_selected=args.model_selected,
        tokenizer_selected=args.tokenizer_selected,
        batch_size=args.batch_size,
        hyperparameter=config,
        re_train=args.re_train,
        grid_search=args.grid_search
    )
