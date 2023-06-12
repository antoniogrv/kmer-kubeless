from typing import Final
from typing import Dict

import numpy as np
import argparse
import logging
import torch
import os

from train_gene_classifier import main as train_gene_classifier

from utils import SEPARATOR
from utils import str2bool
from utils import create_test_name
from utils import test_check
from utils import delete_checkpoints
from utils import create_folders
from utils import setup_logger
from utils import save_result
from utils import close_loggers

TASK: Final = 'gene_fusion'


def main(
        len_read: int,
        len_overlap: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        batch_size: int,
        gc_model_selected: str,
        gc_hyperparameter: Dict[str, any],
        model_selected: str,
        hyperparameter: Dict[str, any],
        grid_search: bool
):

    """
    # generate test name
    test_name: str = create_test_name(
        len_read=len_read,
        len_overlap=len_overlap,
        len_kmer=len_kmer,
        n_words=n_words,
        tokenizer_selected=tokenizer_selected,
        hyperparameter={**gc_hyperparameter, **hyperparameter}
    )

    # check if gene classifier is trained, otherwise train it
    train_gene_classifier(
        len_read=len_read,
        len_overlap=len_overlap,
        len_kmer=len_kmer,
        n_words=n_words,
        model_selected=gc_model_selected,
        tokenizer_selected=tokenizer_selected,
        batch_size=batch_size,
        hyperparameter=gc_hyperparameter,
        grid_search=True
    )
    """


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
    parser.add_argument('-tokenizer_selected', dest='tokenizer_selected', action='store',
                        type=str, default='dna_bert', help='select the tokenizer to be used')

    # gene classifier parameters
    parser.add_argument('-gc_model_selected', dest='gc_model_selected', action='store',
                        type=str, default='dna_bert', help='select the model to be used')
    parser.add_argument('-gc_hidden_size', dest='gc_hidden_size', action='store',
                        type=int, default=768, help='define number of hidden channels')
    parser.add_argument('-gc_n_attention_heads', dest='gc_n_attention_heads', action='store',
                        type=int, default=1, help='define number of attention heads')
    parser.add_argument('-gc_n_beams', dest='gc_n_beams', action='store',
                        type=int, default=1, help='define number of beams')
    parser.add_argument('-gc_n_hidden_layers', dest='gc_n_hidden_layers', action='store',
                        type=int, default=9, help='define number of hidden layers')
    parser.add_argument('-gc_rnn', dest='gc_rnn', action='store',
                        type=str, default='lstm', help='define type of recurrent layer')
    parser.add_argument('-gc_n_rnn_layers', dest='gc_n_rnn_layers', action='store',
                        type=int, default=3, help='define number of recurrent layers')

    # fusion classifier parameters
    parser.add_argument('-batch_size', dest='batch_size', action='store',
                        type=int, default=256, help='define batch size')
    parser.add_argument('-dropout', dest='dropout', action='store',
                        type=float, default=0.5, help='define value of dropout probability')

    # grid search flag
    parser.add_argument('-grid_search', dest='grid_search', action='store', type=str2bool,
                        default=False, help='set true if this script is launching from grid_search script')

    args = parser.parse_args()

    # check tokenizer selected
    if args.tokenizer_selected not in ['dna_bert', 'dna_bert_n']:
        raise Exception('select one of these recurrent layers: ["dna_bert", "dna_bert_n"]')

    # check gene model selected
    if args.gc_model_selected not in ['dna_bert']:
        raise Exception('select one of these recurrent layers: ["dna_bert"]')

    # check gene recurrent layer selected
    if args.gc_rnn not in ['lstm', 'gru']:
        raise Exception('select one of these recurrent layers: ["lstm", "gru"]')

    # init hyperparameter
    gc_config: Dict[str, any] = {
        'hidden_size': args.gc_hidden_size,
        'dropout': args.dropout,
        'n_attention_heads': args.gc_n_attention_heads,
        'n_beams': args.gc_n_beams,
        'n_hidden_layers': args.gc_n_hidden_layers,
        'rnn': args.gc_rnn,
        'n_rnn_layers': args.gc_n_rnn_layers,
    }

    main(
        len_read=args.len_read,
        len_overlap=args.len_overlap,
        len_kmer=args.len_kmer,
        n_words=args.n_words,
        tokenizer_selected=args.tokenizer_selected,
        batch_size=args.batch_size,
        gc_model_selected=args.gc_model_selected,
        gc_hyperparameter=gc_config,
        model_selected='',
        hyperparameter={},
        grid_search=args.grid_search
    )
