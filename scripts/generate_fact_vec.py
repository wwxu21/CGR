import numpy as np
from transformers import AutoConfig, AutoTokenizer
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
from tqdm import tqdm
import faiss
import torch
import os
import sys
sys.path.append("./..")
from CGR.retriever_utils import read_qa, read_comm, get_vec, read_gold
from CGR.retriever import Retriever
def load_model():
    model_name = "sentence-transformers/bert-base-nli-cls-token"
    config_class, model_class, tokenizer_class = AutoConfig, Retriever, AutoTokenizer

    config = config_class.from_pretrained(
        model_name,
        cache_dir="/projdata11/info_fil/wwxu/WorkSpace/OpenBook-dv/cache_file/",
        finetuning_task='fact_encoder',
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        cache_dir="/projdata11/info_fil/wwxu/WorkSpace/OpenBook-dv/cache_file/",
        do_lower_case=True,
    )
    model = model_class.from_pretrained(
        model_name,
        cache_dir="/projdata11/info_fil/wwxu/WorkSpace/OpenBook-dv/cache_file/",
        from_tf=bool(".ckpt" in model_name),
        config=config,
    )
    return config, tokenizer, model

def init_index(db, d):
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)
    index.add(db)    # add vectors to the index
        # actual search
    return index

def batch_search(query_vec, data_vec, vec_size, step, k):
    logger.info('faiss search begin')
    D_list = []
    I_list = []
    for i in tqdm(range(0, data_vec.shape[0], step), desc="Batch Search Iteration"):
        batch_core = data_vec[i:i+step]
        index = init_index(batch_core, vec_size)
        D, I = index.search(query_vec, k)
        D_list.append(D)
        I_list.append(I+i)
    D_array = np.concatenate(D_list, axis=1)
    I_array = np.concatenate(I_list, axis=1)
    topk_pos_row = np.expand_dims(np.arange(D_array.shape[0]),1).repeat(k,axis=1)
    topk_pos_col = np.argsort(-D_array, axis=1)[:,:k]
    topk_index = I_array[topk_pos_row, topk_pos_col]
    return topk_index

def sample_topk(query, data, data_vec, data_save, vec_save):
    sample_index = list(set(query.reshape(-1)))
    sample_data = [data[i] for i in sample_index]
    sample_data_vec = np.asarray([data_vec[i] for i in sample_index])
    comm_dict = {x: sample_data_vec[i_x] for i_x, x in enumerate(sample_data)} # clean duplicated
    sample_data = list(comm_dict.keys())
    sample_data_vec = np.asarray(list(comm_dict.values()))
    np.save(vec_save, sample_data_vec)
    with open(data_save, 'w') as writer:
        for one in sample_data:
            writer.write(one + '\n')

def load_vec(f_addr, vec_addr):
    if f_addr.endswith('jsonl'):
        f = read_gold(f_addr)
    elif f_addr.endswith('txt'):
        f = read_comm(f_addr)
    if os.path.exists(vec_addr):
        logger.info("load vec from disk")
        vec = np.load(vec_addr)
    else:
        logger.info("generate vec")
        vec = get_vec(f, tokenizer, model)
        np.save(vec_addr, vec)
    return f, vec

def prepare_hypo_vec(f_addr):
    f = read_qa(f_addr)
    hypos = [choice['hypo'] for data_one in f for choice in data_one['question']['choices']]
    hypos_vec = get_vec(hypos, tokenizer, model)
    return hypos_vec

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='obqa', type=str, help="which task to perform")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    config, tokenizer, model = load_model()
    vec_size = config.hidden_size

    prefix = './Data/begin/' + args.task

    # # fact
    core_file_addr = './Data/OpenBook/openbook.txt' #the openbook
    core_vec_addr = os.path.join(prefix, 'core.npy')
    core_f, core_vec = load_vec(core_file_addr, core_vec_addr)
    comm_file_addr = './Data/OpenBook/ARC_Clean.txt' #ARC corpus
    comm_vec_addr = os.path.join(prefix, 'obqa.npy')
    comm_f, comm_vec = load_vec(comm_file_addr, comm_vec_addr)

    assert len(comm_f) == comm_vec.shape[0]
