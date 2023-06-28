from typing import Final
from typing import Tuple
from typing import List
from typing import Dict

from multiprocessing.pool import Pool
from functools import partial

import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data.dataset import T_co

from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.model_selection import train_test_split

from dataset import MyDataset

from dataset._concurrent import split_dataset_on_processes
from dataset._concurrent import generate_kmers_from_dataset
from dataset._tools import fusion_simulator
from dataset._tools import art_illumina
from dataset._tools import generate_reads

GENES_PANEL_PATH: Final = os.path.join(os.getcwd(), 'data', 'genes_panel.txt')


class FusionDataset(MyDataset):
    def __init__(
            self,
            root_dir: str,
            conf: Dict[str, any],
            dataset_type: str
    ):
        # call super class
        super().__init__(
            root_dir=root_dir,
            check_dir_name='check',
            check_dict_name='fusion_dataset',
            conf=conf,
            dataset_type=dataset_type
        )

        # ======================== Load Gene Panel ======================== #
        with open(GENES_PANEL_PATH, 'r') as gene_panel_file:
            self.__genes_list: List[str] = gene_panel_file.read().split('\n')

        # ===================== Fusion Simulator Step ===================== #
        __fusim_dir: str = os.path.join(self.root_dir, 'fusim')
        __fusim_fasta_format_path: str = os.path.join(__fusim_dir, '{gene}.fasta')
        __fusim_text_format_path: str = os.path.join(__fusim_dir, '{gene}.text')
        # check if fusim step is already done
        fusim_phase_flag: bool = self.check_dir(__fusim_dir) and self.check_file(GENES_PANEL_PATH)
        if not fusim_phase_flag:
            # create directory if it doesn't exist
            if not os.path.exists(__fusim_dir):
                os.makedirs(__fusim_dir)
            # execute fusion simulator
            fusion_simulator(
                fasta_format_path=__fusim_fasta_format_path,
                text_format_path=__fusim_text_format_path,
                n_fusions=self.conf['n_fusion'],
                genes_list=self.__genes_list
            )
            self.update_dir(__fusim_dir)

        # ======================= ART Illumina Step ======================= #
        __art_dir: str = os.path.join(self.root_dir, f'art_{self.conf["len_read"]}')
        __art_base_format_path: str = os.path.join(__art_dir, '{gene}_art')
        # check if generation of reads step is already done
        art_phase_flag: bool = fusim_phase_flag and self.check_dir(__art_dir)
        if not art_phase_flag:
            # create directory if it doesn't exist
            if not os.path.exists(__art_dir):
                os.makedirs(__art_dir)
            art_illumina(
                len_read=self.conf['len_read'],
                fusim_fasta_format_path=__fusim_fasta_format_path,
                art_base_format_path=__art_base_format_path,
                genes_list=self.__genes_list
            )
            self.update_dir(__art_dir)

        # ==================== Generation of reads Step =================== #
        __chimeric_reads_dataset_path: str = os.path.join(self.processed_dir, f'chimeric_{self.conf["len_read"]}.csv')
        # check if chimeric reads dataset is already generated
        generation_reads_phase_flag: bool = art_phase_flag and self.check_dataset(__chimeric_reads_dataset_path)
        if not generation_reads_phase_flag:
            __chimeric_reads_dataset: pd.DataFrame = generate_reads(
                len_read=self.conf['len_read'],
                fusim_text_format_path=__fusim_text_format_path,
                art_base_format_path=__art_base_format_path,
                genes_list=self.__genes_list
            )
            # save dataset as csv
            __chimeric_reads_dataset.to_csv(__chimeric_reads_dataset_path, index=False)
            self.update_dataset(__chimeric_reads_dataset_path)

        # ==================== Generation of kmers Step =================== #
        __chimeric_kmers_dataset_path: str = os.path.join(
            self.processed_dir,
            f'chimeric_{self.conf["len_read"]}_kmer_{self.conf["len_kmer"]}.csv'
        )
        # check if chimeric kmers dataset is already generated
        generation_kmers_phase_flag: bool = (generation_reads_phase_flag and
                                             self.check_dataset(__chimeric_kmers_dataset_path))
        if not generation_kmers_phase_flag:
            # load chimeric reads dataset
            __chimeric_reads_dataset: pd.DataFrame = pd.read_csv(__chimeric_reads_dataset_path)
            # split dataset on processes
            rows_for_each_process: List[Tuple[int, int]] = split_dataset_on_processes(
                __chimeric_reads_dataset,
                os.cpu_count()
            )
            # init chimeric kmers dataset
            __chimeric_kmers_dataset: pd.DataFrame = pd.DataFrame()
            # call generate all kmers from reads on multi processes
            with Pool(os.cpu_count()) as pool:
                results = pool.imap(partial(
                    generate_kmers_from_dataset,
                    dataset=__chimeric_reads_dataset,
                    len_kmer=self.conf["len_kmer"]
                ), rows_for_each_process)
                # append all local dataset to global dataset
                for local_dataset in results:
                    __chimeric_kmers_dataset = pd.concat([__chimeric_kmers_dataset, local_dataset])
            # save global kmers dataset
            __chimeric_kmers_dataset.to_csv(__chimeric_kmers_dataset_path, index=False)
            self.update_dataset(__chimeric_kmers_dataset_path)

        # ============== Generation of train, val, test Step ============== #
        assert self.conf['classification_type'] in ['fusion']
        __train_dataset_path: str = os.path.join(
            self.processed_dir,
            f'chimeric_{self.conf["len_read"]}_'
            f'kmer_{self.conf["len_kmer"]}_'
            f'{self.conf["classification_type"]}_'
            f'train.csv'
        )
        __val_dataset_path: str = os.path.join(
            self.processed_dir,
            f'chimeric_{self.conf["len_read"]}_'
            f'kmer_{self.conf["len_kmer"]}_'
            f'{self.conf["classification_type"]}_'
            f'val.csv'
        )
        __test_dataset_path: str = os.path.join(
            self.processed_dir,
            f'chimeric_{self.conf["len_read"]}_'
            f'kmer_{self.conf["len_kmer"]}_'
            f'{self.conf["classification_type"]}_'
            f'test.csv'
        )
        # check if train, val and test set are already generateds
        generation_sets_phase_flag: bool = generation_kmers_phase_flag and (
                self.check_dataset(__train_dataset_path) and
                self.check_dataset(__val_dataset_path) and
                self.check_dataset(__test_dataset_path)
        )
        if not generation_sets_phase_flag:
            # load chimeric kmers dataset
            __chimeric_kmers_dataset: pd.DataFrame = pd.read_csv(__chimeric_kmers_dataset_path)
            # create labels for dataset by classification_type value
            if self.conf["classification_type"] == 'fusion':
                __chimeric_kmers_dataset['label'] = np.where(
                    __chimeric_kmers_dataset['gene_1'] == __chimeric_kmers_dataset['gene_2'], 0, 1
                )
            # split dataset in train, val and test set
            __train_dataset, __test_dataset = train_test_split(
                __chimeric_kmers_dataset,
                test_size=0.1
            )
            __train_dataset, __val_dataset = train_test_split(
                __train_dataset,
                test_size=0.11
            )
            # group datasets and dataset paths
            __datasets: List[pd.DataFrame] = [
                __train_dataset,
                __val_dataset,
                __test_dataset
            ]
            __dataset_paths: List[str] = [
                __train_dataset_path,
                __val_dataset_path,
                __test_dataset_path
            ]
            # shuffles rows, saves datasets as csv, and updates hashes
            for i in range(3):
                __datasets[i] = __datasets[i].sample(frac=1).reset_index(drop=True)
                __datasets[i].to_csv(__dataset_paths[i], index=False)

        # load dataset
        self.__dataset_path = os.path.join(
            self.processed_dir,
            f'chimeric_{self.conf["len_read"]}_'
            f'kmer_{self.conf["len_kmer"]}_'
            f'{self.conf["classification_type"]}_'
            f'{self.dataset_type}.csv'
        )
        self.__dataset: pd.DataFrame = pd.read_csv(self.__dataset_path)
        self.__status = self.__dataset.groupby('label')['label'].count()

        # ==================== Create inputs for model ==================== #
        assert self.conf['tokenizer'] is not None
        self.__inputs_path: str = os.path.join(
            self.inputs_dir,
            f'chimeric_{self.conf["len_read"]}_'
            f'kmer_{self.conf["len_kmer"]}_'
            f'n_words_{self.conf["n_words"]}_'
            f'{self.conf["classification_type"]}_'
            f'{self.dataset_type}.pkl'
        )
        # check if inputs tensor are already generateds
        generation_inputs_phase: bool = generation_sets_phase_flag and self.check_file(self.__inputs_path)
        if not generation_inputs_phase:
            # get number of kmers and number of sentences
            __n_kmers: int = self.conf['len_read'] - self['len_kmer'] + 1
            __n_sentences: int = __n_kmers - self.conf['n_words'] + 1

        """
        # create inputs for model
        self.inputs = []
        kmers_sequences: List[List[str]] = list(self.dataset.iloc[:, :n_kmers].values)
        for index, kmers_sequence in enumerate(
                tqdm(kmers_sequences, desc='Creating inputs for the model...', total=len(kmers_sequences))):
            # get all sentences of n_words
            sentences: List[str] = [' '.join(kmers_sequence[i:i + self.__n_words]) for i in range(n_sentences)]
            # tokenizing all sentences
            sentences_tokenized = [
                self.__tokenizer.encode_plus(
                    sentence,
                    padding='max-length',
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.__n_words + 2
                ) for sentence in sentences
            ]
            # extract tensor from sentences tokenized
            _input = []
            for sentence_tokenized in sentences_tokenized:
                input_ids = sentence_tokenized['input_ids']
                token_type_ids = sentence_tokenized['token_type_ids']
                attention_mask = sentence_tokenized['attention_mask']

                # zero-pad up to the sequence length.
                padding_length = self.__n_words + 2 - len(input_ids)
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([1] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)

                _input.append(
                    {
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.int),
                        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int),
                    }
                )

            self.inputs.append(
                {
                    'sentences': _input,
                    'label': torch.tensor([self.dataset.at[index, 'label']], dtype=torch.long)
                }
            )

        print('ciao')

    """

    def __getitem__(self, index) -> T_co:
        pass

    @staticmethod
    def create_conf(
            len_read: int = 150,
            len_kmer: int = 6,
            n_fusion: int = 30,
            classification_type: str = 'fusion',
            tokenizer: PreTrainedTokenizer = None
    ) -> Dict[str, any]:
        return {
            'len_read': len_read,
            'len_kmer': len_kmer,
            'n_fusion': n_fusion,
            'tokenizer': tokenizer,
            'classification_type': classification_type
        }
