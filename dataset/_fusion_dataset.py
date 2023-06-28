from typing import Final
from typing import List

import os

import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import T_co

from transformers.tokenization_utils import PreTrainedTokenizer

from dataset._utils import fusion_simulator
from dataset._utils import generate_reads_with_art

GENES_PANEL_PATH: Final = os.path.join(os.getcwd(), 'data', 'genes_panel.txt')


class FusionDataset(Dataset):
    def __init__(
            self,
            root_dir: str = None,
            len_read: int = 150,
            len_kmer: int = 6,
            n_words: int = 30,
            n_fusions: int = 30,
            dataset_type: str = 'train',
            tokenizer: PreTrainedTokenizer = None
    ):
        # init paths
        assert root_dir is not None
        self.__root_dir: str = root_dir
        self.__fusim_dir: str = os.path.join(self.__root_dir, 'fusim')
        self.__fusim_fasta_format_path: str = os.path.join(self.__fusim_dir, '{gene}.fasta')
        self.__fusim_text_format_path: str = os.path.join(self.__fusim_dir, '{gene}.text')
        self.__chimeric_dir: str = os.path.join(self.__root_dir, 'chimeric')
        self.__art_base_format_path: str = os.path.join(self.__chimeric_dir, '{gene}_art')
        self.__processed_dir: str = os.path.join(self.__root_dir, 'processed_dir')
        self.__chimeric_reads_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'chimeric_{len_read}_dataset.csv'
        )
        self.__train_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'chimeric_{len_read}_kmer_{len_kmer}_train.csv'
        )
        self.__val_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'chimeric_{len_read}_kmer_{len_kmer}_val.csv'
        )
        self.__test_dataset_path: str = os.path.join(
            self.__processed_dir,
            f'chimeric_{len_read}_kmer_{len_kmer}_test.csv'
        )

        # check dataset type
        assert dataset_type in ['train', 'val', 'test']
        self.__dataset_type = dataset_type

        # create folders
        for dir_path in [self.__fusim_dir, self.__chimeric_dir, self.__processed_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # get genes panel
        with open(GENES_PANEL_PATH, 'r') as gene_panel_file:
            self.__genes_list: List[str] = gene_panel_file.read().split('\n')

        # generate chimeric reads with Fusion Simulator
        self.__n_fusions = n_fusions
        # check if fusion simulator is already executed
        __flag: bool = False
        for gene in self.__genes_list:
            __flag = __flag or (not os.path.exists(self.__fusim_fasta_format_path.format(gene=gene)) or
                                not os.path.exists(self.__fusim_text_format_path.format(gene=gene)))
            if __flag is True:
                break
        if __flag:
            # execute fusion simulator
            fusion_simulator(
                fasta_format_path=self.__fusim_fasta_format_path,
                text_format_path=self.__fusim_text_format_path,
                n_fusions=self.__n_fusions,
                genes_list=self.__genes_list
            )

        # check if reads has already simulated with ART Illumina
        self.__len_read = len_read
        if not os.path.exists(self.__chimeric_reads_dataset_path):
            # generate a chimeric reads dataset
            chimeric_reads_dataset: pd.DataFrame = generate_reads_with_art(
                len_read=self.__len_read,
                fasta_format_path=self.__fusim_fasta_format_path,
                text_format_path=self.__fusim_text_format_path,
                art_base_format_path=self.__art_base_format_path,
                n_fusions=self.__n_fusions,
                genes_list=self.__genes_list
            )
            # save dataset as csv
            chimeric_reads_dataset.to_csv(self.__chimeric_reads_dataset_path, index=False)

    def __getitem__(self, index) -> T_co:
        pass
