# Index

## Introduction

## Requirements

Install DNABert form GitHub repository.

```shell
git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
cd ../..

mv DNABERT/src/transformers ./transformers
rm -r DNABERT
```

Install other packages.

```shell
CUDA_VERSION=cu118
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
conda install -c conda-forge biopython
pip install tabulate

apt install genometools
```

## Gene classifier model
The first model that is defined within this project is the **gene classifier** model. 
The goal of this model is to correctly classify sentences in the source gene. 
More formally, we define a sentence as a sentence consisting of *n* words, where each word is a *kmer*.

Starting with a read, we generate all possible kmers, of length ```len_kmer```, of the read. 
Let ```n_words``` be the number of kmers that make up a sentence, then all possible subsets of consecutive 
kmers of cardinality ```n_words``` are generated. This allows all possible sentences to be generated from a 
read. The goal of the classifier is to correctly classify a sentence to the source gene of the read used 
to generate the sentence.

The following are the input parameters of the ```train_gene_classifier.py``` script, which is the script responsible 
for training and evaluating the gene classification model.

### Parameters
* ```len_read```: define length of reads.
* ```len_overlap```: define length of overlapping between reads.
* ```len_kmer```: define length of kmers.
* ```n_words```: number of kmers inside a sentence
* ```tokenizer_selected```: select the tokenizer to be used. The possible values are:
  * *dna_bert*: vocabulary encoding all possible kmer on the alphabet {A, C, G, T}.
  * *dna_bert_n*: vocabulary encoding all possible kmer on the alphabet {A, C, G, T, N}.
* ```batch_size```: define batch size.
* ```model_selected```: defines the model to be tested. Possible inputs are:
  * *dna_bert*: uses the dna bert model to make the classification.
* ```hidden_size```: define number of hidden channels.
* ```n_hidden_layers```: define number of hidden layers.
* ```rnn```: define type of recurrent layer. The possible value are:
  * *lstm*: selects as recurrent layers of type Long short-term memory.
  * *gru*: selects Gated recurrent unit as a recurrent layer.
* ```n_rnn_layers```: define number of recurrent layers.
* ```n_attention_heads```: define number of attention heads.
* ```n_beams```: define number of beams.
* ```dropout```: define value of dropout probability.
* ```grid_search```: if the value is set to true then if the model has already been tested with the 
hyperparameters given as input then the program terminates, otherwise the model is evaluated again on the test set.

Below is the command to train the gene classification model with the best configuration found.
```shell
python3 train_gene_classifier.py -len_read 150 \
                                 -len_overlap 0 \
                                 -len_kmer 6 \
                                 -n_words 30 \
                                 -batch 512 \
                                 -tokenizer_selected dna_bert_n \
                                 -model_selected dna_bert \
                                 -hidden_size 768 \
                                 -n_hidden_layers 7 \
                                 -rnn lstm \
                                 -n_rnn_layers 3 \
                                 -n_attention_heads 1 \
                                 -n_beams 1 \
                                 -dropout 0.5
```

## Gene fusion model