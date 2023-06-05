from typing import List
from typing import Dict
from typing import Any

import pandas as pd
import os

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def save_result(
        result_csv_path: str,
        len_read: int,
        len_overlap: int,
        len_kmer: int,
        n_words: int,
        tokenizer_selected: str,
        hyperparameter: Dict[str, Any],
        y_true: List[int],
        y_pred: List[int]
):
    # init columns of result df
    columns = ['len_read', 'len_overlap', 'len_kmer', 'n_words', 'tokenizer_selected']
    columns += list(hyperparameter.keys())
    columns += ['accuracy', 'precision', 'recall', 'f1-score']

    # create row of df
    values = [len_read, len_overlap, len_kmer, n_words, tokenizer_selected]
    values += [hyperparameter[p] for p in hyperparameter.keys()]
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=1
    )
    values += [accuracy, precision, recall, f_score]
    result_csv: pd.DataFrame = pd.DataFrame(
        [
            values
        ],
        columns=columns
    )

    # check if result dataset exists
    if os.path.exists(result_csv_path):
        global_results_csv: pd.DataFrame = pd.read_csv(result_csv_path)
        global_results_csv = pd.concat([global_results_csv, result_csv])
        global_results_csv = global_results_csv.sort_values(by=['accuracy'], ascending=False)
        global_results_csv.to_csv(result_csv_path, index=False)
    else:
        result_csv.to_csv(result_csv_path, index=False)
