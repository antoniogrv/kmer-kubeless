from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
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


def fusion_simulator(
        fasta_format_path: str,
        text_format_path: str,
        n_fusions: int,
        genes_list: List[str]
) -> None:
    # init path of fusion simulator directory
    fusion_simulator_dir: str = os.path.join(os.getcwd(), 'fusim-0.2.2')
    for gene in tqdm(genes_list, desc=f'Execute Fusion Simulator...', total=len(genes_list)):
        genes_list_tmp: List[str] = genes_list.copy()
        genes_list_tmp.remove(gene)
        # init command
        command: str = f'java -jar {os.path.join(fusion_simulator_dir, "fusim.jar")} ' \
                       f'-g {os.path.join(fusion_simulator_dir, "refFlat.txt")} ' \
                       f'-r {os.path.join(fusion_simulator_dir, "hg19.fa")} ' \
                       f'-f {fasta_format_path.format(gene=gene)}_tmp ' \
                       f'-t {text_format_path.format(gene=gene)} ' \
                       f'-n {n_fusions} ' \
                       f'-1 {gene} ' \
                       f'-2 {",".join(genes_list_tmp)} ' \
                       f'--cds-only ' \
                       f'--auto-correct-orientation 2>/dev/null'
        # execute command
        os.system(command)

        # add a number to the transcript identifier
        fasta_output_tmp_file = SeqIO.parse(open(f'{fasta_format_path.format(gene=gene)}_tmp'), 'fasta')
        fasta_output_file = open(f'{fasta_format_path.format(gene=gene)}', 'w')
        for idx, transcript_tmp in enumerate(fasta_output_tmp_file):
            transcript: SeqRecord = transcript_tmp
            transcript.id = f'{idx}_{transcript.id}'
            SeqIO.write(transcript, fasta_output_file, 'fasta')
        # close all files and remove tmp file
        fasta_output_file.close()
        os.system(f'rm {fasta_format_path.format(gene=gene)}_tmp')


def generate_reads_with_art(
        len_read: int,
        fasta_format_path: str,
        text_format_path: str,
        art_base_format_path: str,
        n_fusions: int,
        genes_list: List[str]
) -> pd.DataFrame:
    # init dataset
    dataset: pd.DataFrame = pd.DataFrame()
    columns: List[str] = ['read', 'gene_1', 'gene_2', 'breakpoint']
    # execute ART on all file generated by Fusion Simulator
    for gene in tqdm(genes_list, desc=f'Generate reads with ART...', total=len(genes_list)):
        # init command
        command: str = f'art_illumina -i {fasta_format_path.format(gene=gene)} ' \
                       f'-l {len_read} ' \
                       f'-f 10 ' \
                       f'-p ' \
                       f'-m 400 ' \
                       f'-s 10 ' \
                       f'-o {art_base_format_path.format(gene=gene)} 1>/dev/null'
        # execute ART Illumina
        os.system(command)

        # open files
        art_aln_output_file = open(f'{art_base_format_path.format(gene=gene)}1.aln', 'r')
        art_fq_output_file = open(f'{art_base_format_path.format(gene=gene)}1.fq', 'r')
        fusim_fasta_output_file = open(fasta_format_path.format(gene=gene), 'r')
        fusim_text_output_file = open(text_format_path.format(gene=gene), 'r')

        # extract the information from the header of the file
        for _ in range(2):
            next(art_aln_output_file)

        print('Here')

        """
        # for each line of ART output file
        while True:
            # read a line of ART file output
            line = art_output_file.readline()
            # check if EOF is reached
            if line == '':
                break
            # get read identification list
            read_identification_list: List[str] = line.split('\t')
            gene_1_fusion_information: List[str] = fusim_text_output_file.readline().split('\t')
            gene_2_fusion_information: List[str] = fusim_text_output_file.readline().split('\t')
            # get information of 1st gene name and 2nd gene name of starting transcript
            name_1_transcript: str = gene_1_fusion_information[2]
            name_2_transcript: str = gene_2_fusion_information[2]
            # check if fusion match
            header: str = f'>ref|{name_1_transcript}-{name_2_transcript}'
            while read_identification_list[0] != header:
                # skip 2 row on text output file
                gene_1_fusion_information: List[str] = fusim_text_output_file.readline().split('\t')
                gene_2_fusion_information: List[str] = fusim_text_output_file.readline().split('\t')
                # skip 2 row on fasta output file
                fusim_fasta_output_file.readline()
                fusim_fasta_output_file.readline()
                # get new information and new header
                name_1_transcript: str = gene_1_fusion_information[2]
                name_2_transcript: str = gene_2_fusion_information[2]
                header: str = f'>ref|{name_1_transcript}-{name_2_transcript}'

            # get len of first gene and second gene of this transcript and transcript
            len_gene_1: int = int(gene_1_fusion_information[6])
            len_gene_2: int = int(gene_2_fusion_information[6])
            len_transcript: int = len_gene_1 + len_gene_2
            transcript: str = fusim_fasta_output_file.readline().strip()

            # get information of reads generated by this transcript
            n_reads: int = int(int(read_identification_list[1][len(header):-4]) / 2)
            # iterate all reads generated by this transcript
            for idx in range(n_reads):
                # get information of strand
                strand: str = read_identification_list[3].strip()
                # get index of start and finish of read inside the transcript
                start_index: int = int(read_identification_list[2])
                # whether the generated read is the reverse and complement of the source read
                if strand == '-':
                    start_index: int = len_transcript - start_index - len_read
                end_index: int = start_index + len_read
                # get value of 1-st gene and 2-nd gene of read and breakpoint
                bp_breakpoint: int = 0
                if start_index < len_gene_1:
                    gene_1: str = gene_1_fusion_information[1]
                    if end_index <= len_gene_1:
                        gene_2: str = gene_1_fusion_information[1]
                    # in fact, there is a fusion
                    else:
                        gene_2: str = gene_2_fusion_information[1]
                        bp_breakpoint = len_gene_1 - start_index + 1
                else:
                    gene_1: str = gene_2_fusion_information[1]
                    gene_2: str = gene_2_fusion_information[1]

                # get and check read
                read: str = art_output_file.readline().strip().upper()
                if len(read) != 150:
                    read: str = read[0:150]
                if '-' in read:
                    read: str = art_output_file.readline().strip().upper()
                else:
                    art_output_file.readline()
                if strand == '+':
                    if read != transcript[start_index:end_index].upper():
                        print('Here')
                elif strand == '-':
                    if Seq(read).reverse_complement() != transcript[start_index:end_index].upper():
                        print('Here')
                # create row of dataset
                row_dataset: pd.DataFrame = pd.DataFrame(
                    [[
                        read,
                        gene_1,
                        gene_2,
                        bp_breakpoint
                    ]], columns=columns
                )
                # append row on local dataset
                dataset = pd.concat([dataset, row_dataset])

                # skip line
                if idx + 1 < n_reads:
                    read_identification_list: List[str] = art_output_file.readline().split('\t')

            # skip line
            fusim_fasta_output_file.readline()
        """
        # close Fusion Simulator output files
        fusim_fasta_output_file.close()
        fusim_text_output_file.close()

    return dataset
