from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

from Bio import SeqIO
import pandas as pd
import os


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
