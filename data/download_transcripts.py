from typing import Final
from typing import Dict
from typing import List

import requests
import sys
import os

"""
1 - Rest API Ensemble.org https://rest.ensembl.org
2 - GET xrefs/symbol/:species/:symbol https://rest.ensembl.org/documentation/info/xref_external
3 - GET sequence/id/:id https://rest.ensembl.org/documentation/info/sequence_id
"""

GENES_PANEL_PATH: Final = os.path.join(os.getcwd(), 'data', 'genes_panel.txt')
TRANSCRIPTS_DIR_PATH: Final = os.path.join(os.getcwd(), 'data', 'transcripts')
SERVER: Final = "https://rest.ensembl.org"


def get_id(gene_value: str) -> List[str]:
    # init query
    ext = f'/xrefs/symbol/homo_sapiens/{gene_value}?object_type=gene'
    # get result
    r = requests.get(
        SERVER + ext,
        headers={
            'Content-Type': 'application/json'
        })
    # if response status is not okay
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    # init list of results
    id_list: List[str] = []
    data_response = r.json()
    # add each result in id_list array
    for data in data_response:
        id_list.append(data['id'])

    return id_list


def get_sequences(id_sequence_value: str):
    # init query
    ext = f'/sequence/id/{id_sequence_value}?type=cdna;multiple_sequences=1'
    # get result
    r = requests.get(
        SERVER + ext,
        headers={
            'Content-Type': 'text/x-fasta'
        })
    # if response status is not okay
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    # return results
    return r.text


def create_file(gene_value, sequence_value):
    with open(os.path.join(TRANSCRIPTS_DIR_PATH, f'{gene_value}-output.fastq'), 'a') as fasta_file:
        fasta_file.write(sequence_value)


if __name__ == '__main__':
    gene_dict: Dict[str, List[str]] = {}

    # check if transcripts dir exists
    if not os.path.exists(TRANSCRIPTS_DIR_PATH):
        os.makedirs(TRANSCRIPTS_DIR_PATH)

    with open(GENES_PANEL_PATH, 'r') as genes_panel_file:
        for gene in genes_panel_file:
            # remove new line character
            gene: str = gene.rstrip('\n')
            gene_dict[gene] = get_id(gene)
            for sequence_id in gene_dict[gene]:
                sequence = get_sequences(sequence_id)
                create_file(gene, sequence)
