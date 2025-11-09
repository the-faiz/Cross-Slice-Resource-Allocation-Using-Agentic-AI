import pandas as pd

from utils.logger_config import get_logger
logger = get_logger("DATASET LOADER")

def load_vnf_data(path):
    logger.info("Loading Dataset")
    df = pd.read_csv(path)
    # Prepare catalog
    vnfs_list = sorted(df['vnf'].unique().tolist())
    vnf_to_models = {}
    for v in vnfs_list:
        sub = df[df['vnf'] == v]
        models = sub.to_dict('records')
        vnf_to_models[v] = models
    NUM_VNFS = len(vnfs_list)
    NUM_MODELS = len(vnf_to_models[vnfs_list[0]])  # TODO : Each VNF can have different number of models
    logger.info(f"NUM_VNFS = {NUM_VNFS} , NUM_MODELS = {NUM_MODELS}")
    logger.info("Dataset Loaded Successfully")

    return df, vnfs_list, vnf_to_models
