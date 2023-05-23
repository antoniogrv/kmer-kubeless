import os

from transformers import DNATokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


class DNABertTokenizer:
    def __init__(
            self,
            model_name_or_path: str = os.path.join(os.getcwd(), '6-new-12w-0'),
            tokenizer_name: str = 'dna6',
            do_lower_case: bool = False,
            cache_dir: str = ''
    ):
        self.__tokenizer: PreTrainedTokenizer = DNATokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            do_lower_case=do_lower_case,
            cache_dir=cache_dir if cache_dir else None,
        )

    @property
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.__tokenizer
