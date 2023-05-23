from typing import Union
from typing import Tuple
from typing import final
from typing import Dict
from typing import List

from tabulate import tabulate
from glob import glob
from Bio import SeqIO
import pandas as pd
import numpy as np
import pickle
import os

from multiprocessing.pool import Pool
from functools import partial

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import T_co

from transformers.tokenization_utils import PreTrainedTokenizer

DATA_DIR: final = os.path.join(os.getcwd(), 'data')
TRANSCRIPTS_DIR: final = os.path.join(DATA_DIR, 'transcripts')
READS_DIR: final = os.path.join(DATA_DIR, 'reads')


def split_reads_file_on_processes(
        reads_files: List[str],
        n_proc: int
) -> List[List[str]]:
    # get number of files and number of files for each process
    n_files: int = len(reads_files)
    n_files_for_process: int = n_files // n_proc
    rest: int = n_files % n_proc

    # split files on different process
    fasta_files_for_each_process: List[List[str]] = []
    rest_added: int = 0
    for i in range(n_proc):
        start: int = i * n_files_for_process + rest_added
        if rest > i:
            end: int = start + n_files_for_process + 1
            fasta_files_for_each_process.append(reads_files[start:end])
            rest_added += 1
        else:
            end: int = start + n_files_for_process
            fasta_files_for_each_process.append(reads_files[start:end])

    return fasta_files_for_each_process


def split_dataset_on_processes(
        dataset: pd.DataFrame,
        n_proc: int
) -> List[Tuple[int, int]]:
    # get number of rows for each process
    n_rows: int = len(dataset)
    n_rows_for_process: int = n_rows // n_proc
    rest: int = n_rows % n_proc

    # split files on different process
    rows_for_each_process: List[(int, int)] = []
    rest_added: int = 0
    for i in range(n_proc):
        start: int = i * n_rows_for_process + rest_added
        if rest > i:
            end: int = start + n_rows_for_process + 1
            rows_for_each_process.append((start, end))
            rest_added += 1
        else:
            end: int = start + n_rows_for_process
            rows_for_each_process.append((start, end))

    return rows_for_each_process


def generate_kmers_from_sequences(
        reads_files: List[str],
        dir_path: str,
        len_read: str,
        len_overlap: str,
        len_kmer: int,
        labels: Dict[str, int]
) -> pd.DataFrame:
    # init dataset
    dataset: pd.DataFrame = pd.DataFrame()
    # for each read file
    for reads_file in reads_files:
        # open file with SeqIO
        fasta_file = SeqIO.parse(open(
            os.path.join(dir_path, f'{reads_file}_{len_read}_{len_overlap}.reads')
        ), 'fasta')
        # get kmers of all read of file
        for reads in fasta_file:
            sequence: str = reads.seq
            columns: List[str] = []
            values: List[Union[str, int]] = []
            n_kmers: int = len(sequence) + 1 - len_kmer
            for i in range(n_kmers):
                columns.append(f'k_{i}')
                values.append(sequence[i:i + len_kmer].__str__())
            # append label value
            values.append(labels[reads_file])
            columns.append('label')
            # create row by values
            row_dataset: pd.DataFrame = pd.DataFrame(
                [
                    values
                ], columns=columns
            )
            # append row on local dataset
            dataset = pd.concat([dataset, row_dataset])

    return dataset


def generate_sentences_from_kmers(
        rows_index: Tuple[int, int],
        dataset: pd.DataFrame,
        n_words: int
) -> pd.DataFrame:
    # init dataset
    sentences_dataset: pd.DataFrame = pd.DataFrame()
    # get start and end indexes
    start, end = rows_index
    for index in range(start, end):
        # get row of dataset with index
        row: pd.DataFrame = dataset.iloc[[index]]
        # drop NaN values
        row = row.dropna(axis='columns')
        # get kmers and label from row
        kmers: List[str] = list(row.values[0][:-1])
        label: int = int(row.values[0][-1])
        # generate sentences
        n_sentences: int = len(kmers) + 1 - n_words
        if n_sentences < 1:
            continue
        for i in range(n_sentences):
            sentence = ' '.join(kmers[i:i + n_words])
            row_dataset: pd.DataFrame = pd.DataFrame(
                [[
                    sentence,
                    label
                ]], columns=['sentence', 'label']
            )
            # append row on local dataset
            sentences_dataset = pd.concat([sentences_dataset, row_dataset])

    return sentences_dataset


class DNADataset(Dataset):
    def __init__(
            self,
            root_dir: str = None,
            len_read: int = 150,
            len_overlap: int = 0,
            len_kmer: int = 6,
            n_words: int = 20,
            dataset_type: str = 'train',
            tokenizer: PreTrainedTokenizer = None
    ):
        # init paths
        assert root_dir is not None
        self.__root_dir: str = root_dir
        self.__transcripts_dir: str = os.path.join(self.__root_dir, 'transcripts')
        self.__reads_dir: str = os.path.join(self.__root_dir, 'reads')
        self.__processed_dir: str = os.path.join(self.__root_dir, 'processed_dir')
        self.__labels_dict_path: str = os.path.join(self.__processed_dir, 'labels.pkl')
        self.__kmers_dataset_path: str = os.path.join(self.__processed_dir, f'kmer_{len_kmer}.csv')
        self.__train_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'kmer_{len_kmer}_n_words_{n_words}_train.csv'
        )
        self.__val_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'kmer_{len_kmer}_n_words_{n_words}_val.csv'
        )
        self.__test_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'kmer_{len_kmer}_n_words_{n_words}_test.csv'
        )

        # check dataset type
        assert dataset_type in ['train', 'val', 'test']
        self.__dataset_type = dataset_type

        # create folders
        for dir_path in [self.__reads_dir, self.__processed_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # check if labels map was create
        if os.path.exists(self.__labels_dict_path):
            with open(self.__labels_dict_path, 'rb') as handle:
                self.__labels: Dict[str, int] = pickle.load(handle)
                self.__n_classes: int = len(self.__labels.keys())
        else:
            # gets the genes panel
            self.__labels: Dict[str, int] = {}
            self.__n_classes: int = 0
            for file_path in glob(os.path.join(self.__transcripts_dir, '*.fastq')):
                file_name: str = os.path.basename(file_path)
                gene_name: str = file_name[:file_name.index('-output')]
                self.__labels[gene_name] = self.__n_classes
                self.__n_classes += 1
            with open(self.__labels_dict_path, 'wb') as handle:
                pickle.dump(self.__labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # generate reads from transcripts
        self.__len_read: int = len_read
        self.__len_overlap: int = len_overlap
        for file_path in glob(os.path.join(self.__transcripts_dir, '*.fastq')):
            file_name: str = os.path.basename(file_path)
            gene_name: str = file_name[:file_name.index('-output')]
            reads_file_name: str = f'{gene_name}_{self.__len_read}_{self.__len_overlap}.reads'
            reads_file_path: str = os.path.join(self.__reads_dir, reads_file_name)
            if not os.path.exists(reads_file_path):
                # define and execute command
                command: str = f'gt shredder ' \
                               f'-minlength {self.__len_read} ' \
                               f'-maxlength {self.__len_read} ' \
                               f'-overlap {self.__len_overlap} ' \
                               f'-clipdesc no ' \
                               f'{file_path} > {reads_file_path}'
                os.system(command)
                # remove generated files
                for file_ext in ['.sds', '.ois', '.md5', '.esq', '.des', '.ssp']:
                    os.system(f'rm {file_path}{file_ext}')

        # create kmers datasets if it doesn't exist
        self.__len_kmer: int = len_kmer
        if not os.path.exists(self.__kmers_dataset_path):
            reads_files_for_each_process: List[List[str]] = split_reads_file_on_processes(
                reads_files=list(self.__labels.keys()),
                n_proc=os.cpu_count()
            )
            # create global dataset
            n_kmers: int = self.__len_read + 1 - self.__len_kmer
            columns: List[str] = [f'k_{i}' for i in range(n_kmers)]
            columns.append('label')
            self.__kmers_dataset: pd.DataFrame = pd.DataFrame(
                columns=columns
            )
            # call generate kmers from sequence on multi processes
            with Pool(os.cpu_count()) as pool:
                results = pool.imap(partial(
                    generate_kmers_from_sequences,
                    dir_path=self.__reads_dir,
                    len_read=self.__len_read,
                    len_overlap=self.__len_overlap,
                    len_kmer=self.__len_kmer,
                    labels=self.__labels
                ), reads_files_for_each_process)
                # append all local dataset to global dataset
                for result in results:
                    self.__kmers_dataset = pd.concat([self.__kmers_dataset, result])
            self.__kmers_dataset.to_csv(self.__kmers_dataset_path, index=False)
        else:
            self.__kmers_dataset: pd.DataFrame = pd.read_csv(self.__kmers_dataset_path)

        # generate train, val and test set
        self.__n_words: int = n_words
        if (not os.path.exists(self.__train_dataset_path) or
                not os.path.exists(self.__val_dataset_path) or
                not os.path.exists(self.__test_dataset_path)):
            # split dataset in train, val and test set
            train_reads_dataset, test_reads_dataset = train_test_split(
                self.__kmers_dataset,
                test_size=0.1
            )
            train_reads_dataset, val_reads_dataset = train_test_split(
                train_reads_dataset,
                test_size=0.11
            )
            datasets: List[pd.DataFrame] = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
            # generate sentences for each dataset
            for index, reads_dataset in enumerate([train_reads_dataset, val_reads_dataset, test_reads_dataset]):
                reads_dataset.reset_index(drop=True, inplace=True)
                rows_for_each_process: List[Tuple[int, int]] = split_dataset_on_processes(reads_dataset, os.cpu_count())
                # call generate sentences from kmers on multi processes
                with Pool(os.cpu_count()) as pool:
                    results = pool.imap(partial(
                        generate_sentences_from_kmers,
                        dataset=reads_dataset,
                        n_words=self.__n_words
                    ), rows_for_each_process)
                    for result in results:
                        datasets[index] = pd.concat([datasets[index], result])
            # write train, val and test sets
            datasets[0] = datasets[0].sample(frac=1).reset_index(drop=True)
            datasets[0].to_csv(self.__train_dataset_path, index=False)
            datasets[1] = datasets[1].sample(frac=1).reset_index(drop=True)
            datasets[1].to_csv(self.__val_dataset_path, index=False)
            datasets[2] = datasets[2].sample(frac=1).reset_index(drop=True)
            datasets[2].to_csv(self.__test_dataset_path, index=False)

        # load dataset
        self.__dataset_path = os.path.join(
            self.__processed_dir,
            f'kmer_{self.__len_kmer}_n_words_{self.__n_words}_{self.__dataset_type}.csv'
        )
        self.__dataset: pd.DataFrame = pd.read_csv(self.__dataset_path)
        self.__status = self.__dataset.groupby('label')['label'].count()

        # save tokenizer
        assert tokenizer is not None
        self.__tokenizer = tokenizer

        # tokenize all sequences
        self.texts = [
            self.__tokenizer.encode_plus(
                sentence,
                padding='max-length',
                add_special_tokens=True,
                truncation=True,
                max_length=self.__n_words + 2
            ) for sentence in self.__dataset['sentence']
        ]

        # create inputs for model
        self.inputs = []
        for index, text in enumerate(self.texts):
            input_ids = text['input_ids']
            token_type_ids = text['token_type_ids']
            attention_mask = text['attention_mask']

            # zero-pad up to the sequence length.
            padding_length = self.__n_words + 2 - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([1] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            self.inputs.append(
                {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.int),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int),
                    'label': torch.tensor([self.__dataset.at[index, 'label']], dtype=torch.long)
                }
            )

    def classes(self):
        return self.__n_classes

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, index) -> T_co:
        return self.inputs[index]

    @property
    def labels(self):
        return self.__labels

    @property
    def status(self):
        return self.__status

    @property
    def dataset_status(self):
        table: List[List[int, int]] = [[label, record] for label, record in self.__status.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['label', 'no. records'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'
