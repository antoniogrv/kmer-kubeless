from typing import Dict

import numpy as np
import argparse
import logging
import torch
import os

from tokenizer import DNABertTokenizer

from dataset import DNADataset
from torch.utils.data import DataLoader

from model import Model
from model.dna_bert import DNABert

from torch.optim import AdamW
from sklearn.utils import class_weight
from model import train
from model import predict
from sklearn.metrics import classification_report

from utils import SEPARATOR
from utils import create_test_name
from utils import test_check
from utils import create_folders
from utils import setup_logger
from utils import save_result
from utils import close_loggers


def main(
        len_read: int,
        len_overlap: int,
        len_kmer: int,
        n_words: int,
        model_selected: str,
        tokenizer_selected: str,
        batch_size: int,
        hyperparameter: Dict[str, any]
):
    # generate test name
    test_name: str = create_test_name(
        len_read=len_read,
        len_overlap=len_overlap,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter=hyperparameter
    )

    # check if this configuration is already tested
    if not test_check(model_name=model_selected, parent_name=test_name):
        # create folders and get path
        log_path, model_path = create_folders(
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

        # init tokenizer
        tokenizer = None
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

        # load train and validation dataset
        train_dataset = DNADataset(
            root_dir=os.path.join(os.getcwd(), 'data'),
            tokenizer=tokenizer.get_tokenizer,
            dataset_type='train'
        )
        val_dataset = DNADataset(
            root_dir=os.path.join(os.getcwd(), 'data'),
            tokenizer=tokenizer.get_tokenizer,
            dataset_type='val'
        )

        # log information
        logger.info(f'Read len: {len_read}')
        logger.info(f'Overlap len: {len_overlap}')
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
        logger.info(train_dataset.dataset_status)
        logger.info('No. records val set')
        logger.info(val_dataset.dataset_status)

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
        for idx, label in enumerate(train_dataset.status):
            y = np.append(y, [idx] * label)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights: torch.Tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # update hyperparameter
        hyperparameter['vocab_size'] = tokenizer.get_tokenizer.vocab_size
        hyperparameter['n_classes'] = train_dataset.classes()

        # define model
        model: Model = DNABert(
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
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            model_path=model_path,
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
    log_path, model_path = create_folders(model_name=model_selected, parent_name=test_name)
    # init loggers
    logger: logging.Logger = setup_logger('logger', os.path.join(log_path, 'logger.log'))

    # init tokenizer
    tokenizer = None
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

    # load test dataset
    test_dataset = DNADataset(
        root_dir=os.path.join(os.getcwd(), 'data'),
        tokenizer=tokenizer.get_tokenizer,
        dataset_type='test'
    )
    # log test dataset status
    logger.info('No. records test set')
    logger.info(test_dataset.dataset_status)

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
    y_true, y_pred = predict(
        model=model,
        test_loader=test_loader,
        device=device
    )

    # log classification report
    report: str = classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=1,
        target_names=test_dataset.labels.keys()
    )
    logger.info(report)

    # close loggers
    close_loggers([logger])
    del logger

    # save result
    save_result(
        result_csv_path=os.path.join(os.getcwd(), 'log', model_selected, 'results.csv'),
        len_read=len_read,
        len_overlap=len_overlap,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter=model.hyperparameter,
        y_true=y_true,
        y_pred=y_pred
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-len_read', dest='len_read', action='store',
                        type=int, default=150, help='define length of reads')
    parser.add_argument('-len_overlap', dest='len_overlap', action='store',
                        type=int, default=0, help='define length of overlapping between reads')
    parser.add_argument('-len_kmer', dest='len_kmer', action='store',
                        type=int, default=6, help='define length of kmers')
    parser.add_argument('-n_words', dest='n_words', action='store',
                        type=int, default=20, help='number of kmers inside a sentence')
    parser.add_argument('-model_selected', dest='model_selected', action='store',
                        type=str, default='dna_bert', help='select the model to be used')
    parser.add_argument('-tokenizer_selected', dest='tokenizer_selected', action='store',
                        type=str, default='dna_bert', help='select the tokenizer to be used')
    parser.add_argument('-batch_size', dest='batch_size', action='store',
                        type=int, default=256, help='define batch size')
    parser.add_argument('-hidden_size', dest='hidden_size', action='store',
                        type=int, default=768, help='define number of hidden channels')
    parser.add_argument('-dropout', dest='dropout', action='store',
                        type=float, default=0.5, help='define value of dropout probability')
    parser.add_argument('-n_attention_heads', dest='n_attention_heads', action='store',
                        type=int, default=12, help='define number of attention heads')
    parser.add_argument('-n_beams', dest='n_beams', action='store',
                        type=int, default=1, help='define number of beams')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', action='store',
                        type=int, default=12, help='define number of hidden layers')
    parser.add_argument('-rnn', dest='rnn', action='store',
                        type=str, default='lstm', help='define type of recurrent layer')
    parser.add_argument('-n_rnn_layers', dest='n_rnn_layers', action='store',
                        type=int, default=2, help='define number of recurrent layers')

    args = parser.parse_args()

    # check model selected
    if args.model_selected not in ['dna_bert']:
        raise Exception('select one of these recurrent layers: ["dna_bert"]')

    # check tokenizer selected
    if args.tokenizer_selected not in ['dna_bert', 'dna_bert_n']:
        raise Exception('select one of these recurrent layers: ["dna_bert", "dna_bert_n"]')

    # check recurrent layer selected
    if args.rnn not in ['lstm', 'gru']:
        raise Exception('select one of these recurrent layers: ["lstm", "gru"]')

    # init hyperparameter
    config: Dict[str, any] = {
        'hidden_size': args.hidden_size,
        'dropout': args.dropout,
        'n_attention_heads': args.n_attention_heads,
        'n_beams': args.n_beams,
        'n_hidden_layers': args.n_hidden_layers,
        'rnn': args.rnn,
        'n_rnn_layers': args.n_rnn_layers,
    }

    main(
        len_read=args.len_read,
        len_overlap=args.len_overlap,
        len_kmer=args.len_kmer,
        n_words=args.n_words,
        model_selected=args.model_selected,
        tokenizer_selected=args.tokenizer_selected,
        batch_size=args.batch_size,
        hyperparameter=config
    )
