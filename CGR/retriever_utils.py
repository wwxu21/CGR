import faiss
import torch
from tqdm import tqdm
import json
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np

def read_amr(in_file):
    with open(in_file, 'r') as reader:
        out_dict = {}
        for line in reader:
            ob_line = json.loads(line)
            ob_text = ob_line['text']
            out_dict[ob_text] = ob_line['amr']
    return out_dict

def read_qa(addr):
    qa_f = []
    with open(addr, 'r') as reader:
        query_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in query_tqdm:
            json_line = json.loads(line)
            qa_f.append(json_line)
    return qa_f

def read_fact(addr):
    fact_f = []
    with open(addr, 'r') as reader:
        logger.info("reading from {}".format(addr))
        query_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in query_tqdm:
            json_line = json.loads(line)
            fact_f.append(json_line['text'])
    return fact_f

def read_gold(addr):
    fact_f = []
    with open(addr, 'r') as reader:
        logger.info("reading from {}".format(addr))
        query_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in query_tqdm:
            json_line = json.loads(line)
            fact_f.append(json_line['fact1'])
            fact_f.append(json_line['fact2'])
    return fact_f

def read_comm(addr):
    fact_f = []
    with open(addr, 'r') as reader:
        logger.info("reading from {}".format(addr))
        query_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in query_tqdm:
            fact_f.append(line.strip())
    return fact_f

def batchfy(input, tokenizer, batch_size=32, max_len=None, pad=1):
    batches = []
    num, data = 0, []
    for i in range(len(input)):
        num += 1
        data.append(input[i])
        if num >= batch_size:
            batches.append(data)
            num, data = 0, []
    if len(data) != 0:
        batches.append(data)

    for batch in batches:
        batch_id = [tokenizer.encode(x) for x in batch]
        if max_len is None:
            max_size = min(512, max([len(x) for x in batch_id]))
        else:
            max_size = max_len
        batch_id = [x[:max_size] for x in batch_id]
        input_ids = [x + [pad] * (max_size - len(x)) for x in batch_id]
        input_mask = [len(x) * [1] + (max_size - len(x)) * [0] for x in batch_id]
        yield input_ids, input_mask

def get_vec(data, tokenizer, model, batch_size=32, save=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []

    data_iterator = tqdm(batchfy(
        data,
        tokenizer,
        batch_size,
        max_len=None,
        pad=tokenizer.pad_token_id,
    ), desc="Iteration")

    for batch in data_iterator:
        model.eval()
        batch = tuple(torch.tensor(t, dtype=torch.long).to(device) for t in batch)
        with torch.no_grad():
            input = {
                "query_ids": batch[0],
                "query_mask": batch[1],
            }
            output = model(**input)
            outputs.append(output.cpu())
    vectors = torch.cat(outputs, dim=0)
    vectors = vectors.numpy()
    if save is not None:
        np.save(save, vectors)
    return vectors

def init_index(db, d):
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16LookupTables = True
    index = faiss.GpuIndexFlatIP(res, d, config)
    index.add(db)    # add vectors to the index
        # actual search
    return index

def batch_search(query_vec, data_vec, vec_size, step, k):
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
    topk_pos_col = np.argsort(D_array, axis=1)[:,:k]
    topk_I = I_array[topk_pos_row, topk_pos_col]
    topk_D = D_array[topk_pos_row, topk_pos_col]
    return topk_D, topk_I

def statis(input_f):
    gold_hit = 0
    F3_hit = 0
    current_step = 0
    for i_e, example in enumerate(input_f):
        choices = example["question"]['choices']
        gold_fact = example['fact1']
        answerKey = ord(example['answerKey']) - ord('A')
        find = False
        for i_c, choice in enumerate(choices):
            hypo = choice['hypo']
            para_text = choice['para']
            if i_c == answerKey:
                if gold_fact.lower() in para_text:
                    gold_hit += 1
            else:
                if not find and gold_fact.lower() in para_text:
                    F3_hit += 1
                    find = True
        current_step += len(choices)
    return gold_hit / len(input_f), F3_hit / len(input_f)

def combine_search_results(D1, I1, D2, I2, k):
    new_I = []
    for i in range(D1.shape[0]):
        j1, j2 = 0, 0
        new_I_one = {}
        while len(new_I_one) < k:
            if D1[i][j1] <= D2[i][j2] and D1[i][j1] not in new_I_one:
                new_I_one[I1[i][j1]] = 1
                j1 += 1
            elif D1[i][j1] > D2[i][j2] and D1[i][j1] not in new_I_one:
                new_I_one[I2[i][j2]] = 1
                j2 += 1
        new_I.append(list(new_I_one.keys()))
    return np.asarray(new_I)