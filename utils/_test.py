from typing import Dict

import shutil
import os


def create_test_id(
        len_read: int,
        len_kmer: int,
        n_words: int,
        tokenizer: str,
        gc_hyperparameters: Dict[str, any],
        fc_hyperparameters: Dict[str, any] = None
) -> str:
    test_name: str = f'{len_read}_{len_kmer}_{n_words}_{tokenizer}'
    for parameter in gc_hyperparameters.keys():
        value = gc_hyperparameters[parameter]
        if isinstance(value, float):
            value = int(value * 10)
        test_name += f'_{parameter}_{value}'
    if fc_hyperparameters is not None:
        for parameter in fc_hyperparameters.keys():
            value = fc_hyperparameters[parameter]
            if isinstance(value, float):
                value = int(value * 10)
            test_name += f'_{parameter}_{value}'

    return test_name


def init_test(
        result_dir: str,
        task: str,
        model_selected: str,
        test_id: str,
        model_name: str,
        re_train: bool
):
    # get log dir and model dir
    parent_dir: str = os.path.join(result_dir, task, model_selected)
    test_dir: str = os.path.join(parent_dir, test_id)
    log_dir: str = os.path.join(test_dir, 'log')
    model_dir: str = os.path.join(test_dir, 'model')

    # control if you have to retrain
    if re_train:
        for directory in [log_dir, model_dir]:
            shutil.rmtree(directory)

    # create directories
    for directory in [log_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # check if the model has not already been trained
    model_path: str = os.path.join(model_dir, f'{model_name}.h5')

    return parent_dir, test_dir, log_dir, model_dir, model_path
