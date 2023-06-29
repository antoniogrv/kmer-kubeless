from typing import Dict
from typing import Any

import shutil
import os


def create_test_name(
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        hyperparameter: Dict[str, Any]
) -> str:
    test_name: str = f'{len_read}_{len_kmer}_{n_words}_{tokenizer_selected}'
    for parameter in hyperparameter.keys():
        value = hyperparameter[parameter]
        if isinstance(value, float):
            value = int(value * 10)
        test_name += f'_{parameter}_{value}'

    return test_name


def test_check(task: str, model_name: str, parent_name: str) -> bool:
    log_path = os.path.join(os.getcwd(), 'log', task, model_name, parent_name)
    if os.path.exists(log_path):
        model_path = os.path.join(log_path, 'model', 'model.h5')
        if os.path.exists(model_path):
            return True
        else:
            shutil.rmtree(log_path)
            return False
    else:
        return False


def evaluate_check(task: str, model_name: str, parent_name: str) -> bool:
    result_path = os.path.join(os.getcwd(), 'log', task, model_name, parent_name, 'result.log')
    return os.path.exists(result_path)
