from typing import Final
import argparse

# test utils
from utils._test import create_test_name
from utils._test import test_check
from utils._test import evaluate_check

# file and directory utils
from utils._file import create_folders

# metric utils
from utils._metric import save_result

# logging utils
from utils._logger import setup_logger
from utils._logger import close_loggers

SEPARATOR: Final = '\n------------------------------------\n'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
