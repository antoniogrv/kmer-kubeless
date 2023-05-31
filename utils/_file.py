import os


def create_folders(model_name: str, parent_name: str):
    # create log folder
    log_path = os.path.join(os.getcwd(), '../log', model_name, parent_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # create model folder
    model_path = os.path.join(log_path, '../model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return log_path, model_path
