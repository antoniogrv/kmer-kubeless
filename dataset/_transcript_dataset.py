from typing import Hashable
from pandas import Series
from typing import Union
from typing import Tuple
from typing import Dict
from typing import List

from tabulate import tabulate
from tqdm import tqdm
from glob import glob
import pandas as pd
import pickle
import os

from multiprocessing.pool import Pool
from functools import partial

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import T_co

from transformers.tokenization_utils import PreTrainedTokenizer

from dataset._utils import split_dataset_on_processes
from dataset._utils import split_reads_file_on_processes
from dataset._utils import generate_sentences_from_kmers
from dataset._utils import generate_kmers_from_sequences


class TranscriptDataset(Dataset):
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
            ) for sentence in tqdm(
                self.__dataset['sentence'],
                desc=f'Tokenization of the {self.__dataset_type} set...',
                total=len(self.__dataset['sentence'])
            )
        ]

        # create inputs for model
        self.inputs = []
        for index, text in enumerate(tqdm(self.texts, desc='Creating inputs for the model...', total=len(self.texts))):
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
        table: List[List[Union[Hashable, Series]]] = [[label, record] for label, record in self.__status.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['label', 'no. records'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'
