from typing import List

from itertools import product
import collections
import os

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_bert import BasicTokenizer


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class DNABertTokenizer:
    def __init__(
            self,
            root_dir: str,
            len_kmer: int = 6,
            add_n: bool = False,
            do_lower_case: bool = False,
            pad_token: str = '[PAD]',
            unk_token: str = '[UNK]',
            cls_token: str = '[CLS]',
            sep_token: str = '[SEP]',
            mask_token: str = '[MASK]'
    ):
        # define path
        self.__vocab_name = f'kmer_{len_kmer}{"_n" if add_n else ""}'
        self.__vocab_path = os.path.join(root_dir, f'{self.__vocab_name}.txt')

        # check if vocab is defined
        if not os.path.exists(self.__vocab_path):
            # create vocab
            vocabs: List[str] = []
            words: List[str] = ['A', 'T', 'C', 'G', 'N']
            if not add_n:
                words = words[:-1]
            for comb in product(words, repeat=len_kmer):
                vocabs.append(''.join(comb))
            with open(self.__vocab_path, "w") as vocab_file:
                for special_token in [pad_token, unk_token, cls_token, sep_token, mask_token]:
                    vocab_file.write(f'{special_token}\n')
                for vocab in vocabs:
                    vocab_file.write(f'{vocab}\n')

        self.__tokenizer: MyDNATokenizer = MyDNATokenizer(
            vocab_name=self.__vocab_name,
            vocab_file=self.__vocab_path,
            len_kmer=len_kmer,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token
        )

        self.t = self.__tokenizer

    @property
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.__tokenizer


class MyDNATokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BertTokenizer.
    :class:`~transformers.BertTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_basic_tokenize=True
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_basic_tokenize=True
    """

    def save_vocabulary(self, save_directory):
        raise NotImplementedError

    def __init__(
            self,
            vocab_name,
            vocab_file,
            len_kmer: int,
            do_lower_case=False,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
    ):
        """Constructs a BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # save vocab name
        self.__vocab_name = vocab_name

        # take into account special tokens
        self.max_len_single_sentence = self.max_len - 2
        self.max_len_sentences_pair = self.max_len - 3

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.kmer = len_kmer
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
            split_tokens.append(token)
        # print(split_tokens)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return cls + token_ids_0 + sep
            else:
                output = []
                num_pieces = int(len(token_ids_0) // 510) + 1
                for i in range(num_pieces):
                    output.extend(cls + token_ids_0[510 * i:min(len(token_ids_0), 510 * (i + 1))] + sep)
                return output

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        if len(token_ids_0) < 510:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            output = []
            num_pieces = int(len(token_ids_0) // 510) + 1
            for i in range(num_pieces):
                output.extend([1] + ([0] * (min(len(token_ids_0), 510 * (i + 1)) - 510 * i)) + [1])
            return output

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return len(cls + token_ids_0 + sep) * [0]
            else:
                num_pieces = int(len(token_ids_0) // 510) + 1
                return (len(cls + token_ids_0 + sep) + 2 * (num_pieces - 1)) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def __str__(self):
        return self.__vocab_name
