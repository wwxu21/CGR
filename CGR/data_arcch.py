from CGR.utils import convert_examples_to_features, processors
from CGR.retriever_utils import get_vec, init_index, statis, read_amr, read_qa, read_comm, read_fact, batchfy
from CGR.sort_with_AMR import sort_with_AMR
import torch
import math
import os
import numpy as np
import logging
import random
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import json
class AMRDataLoader(object):
    def __init__(self, args, fact_data, model_tokenizer, retriever_tokenizer, evaluate=False, test=False):
        self.model_tokenizer, self.retriever_tokenizer = model_tokenizer, retriever_tokenizer
        self.batch_size = args.train_batch_size
        self.ngpu = int(args.train_batch_size / args.per_gpu_train_batch_size)
        self.max_len = args.max_len
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.kfact = args.kfact
        self.T = args.hop
        self.task_name = args.task_name
        self.train = False
        self.fp_16 = args.fp16
        self.qa_f = {}
        self.max_fact = args.max_fact
        if evaluate:
            self.datasetID = 'dev'
            self.qa_f['dev'] = read_qa(os.path.join(args.data_dir, "dev.jsonl"))
        elif test:
            self.datasetID = 'test'
            self.qa_f['test'] = read_qa(os.path.join(args.data_dir, "test.jsonl"))
        else:
            self.datasetID = 'train'
            self.qa_f['train'] = read_qa(os.path.join(args.data_dir, "train.jsonl"))
            if args.do_eval:
                self.qa_f['dev'] = read_qa(os.path.join(args.data_dir, "dev.jsonl"))
            if args.do_test:
                self.qa_f['test'] = read_qa(os.path.join(args.data_dir, "test.jsonl"))

        if len(fact_data) == 6:
            _ ,_ , _, comm_f, comm_vec, comm_dict = fact_data
            self.fact_dict = {"comm":comm_f}
            self.amr_dict = {"comm":comm_dict}
            self.vec_dict = {"comm":comm_vec}
        else:
            assert False

    def retrieve_fact(self, retriever):
        if self.datasetID == 'train':
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            vec_size = retriever.config.hidden_size
            index_comm = init_index(self.vec_dict['comm'], vec_size)
            qa_post = {}
            for x in self.qa_f:
                qa_post[x] = self.retrieve_fact_faiss(retriever, self.qa_f[x], index_all=index_comm)
                with open(os.path.join(self.output_dir, x + '.jsonl'), "w") as writer:
                    for line in qa_post[x]:
                        writer.write(json.dumps(line) + "\n")
            self.qa_post = qa_post['train']
        else:
            temp_path = os.path.join(self.output_dir, self.datasetID + '.jsonl')
            if os.path.exists(temp_path):
                self.qa_post = read_qa(temp_path)
            else:
                vec_size = retriever.config.hidden_size
                index_comm = init_index(self.vec_dict['comm'], vec_size)
                # hypos_x = [choice['hypo'] for data_one in self.qa_f[self.datasetID] for choice in data_one['question']['choices']]
                self.qa_post = self.retrieve_fact_faiss(retriever, self.qa_f[self.datasetID], index_all=index_comm)

    def retrieve_fact_faiss(self, retriever, qa_f, index_all):
        Ilast = None
        Icomm = []
        t = 0
        index_comm = index_all
        hypos = [choice['hypo'] for data_one in qa_f for choice in data_one['question']['choices']]
        num_example = len(hypos)
        while t < self.T:
            if t == 0:
                query = prepare_query(hypos, self.fact_dict['comm'], Ilast=Ilast)
                query_vec = get_vec(query, self.retriever_tokenizer, retriever, batch_size=64)
                Dcomm_one, Icomm_one = index_comm.search(query_vec, self.kfact)
                Icomm.append(Icomm_one.reshape([num_example, -1]))
                Ilast = Icomm_one
            elif t == 1:
                query = prepare_query(hypos, self.fact_dict['comm'], Ilast=Ilast)
                query_vec = get_vec(query, self.retriever_tokenizer, retriever, batch_size=64)
                Dcomm_one, Icomm_one = index_comm.search(query_vec, self.kfact)
                Ilast = update_DI(Dcomm_one.reshape([num_example, -1]), Icomm_one.reshape([num_example, -1]), self.kfact)
                Icomm.append(Icomm_one.reshape([num_example, -1]))
            else:
                query = prepare_query(hypos, self.fact_dict['comm'], Ilast=Ilast)
                query_vec = get_vec(query, self.retriever_tokenizer, retriever, batch_size=64)
                Dcomm_one, Icomm_one = index_comm.search(query_vec, self.kfact)
                Ilast = update_DI(Dcomm_one.reshape([num_example, -1]), Icomm_one.reshape([num_example, -1]), self.kfact)
                Icomm.append(Icomm_one.reshape([num_example, -1]))
            t += 1
        Icomm = np.concatenate(Icomm, axis=1)
        qa_annotated = []
        current_step = 0

        for i_e, example in enumerate(qa_f):
            choices = example['question']['choices']
            choice_num = len(choices)
            paras = []
            paras_amr = []
            indexs = []
            for i_c, choice in enumerate(choices):
                para_aristo = choice['para']
                comm_index_one_raw = list(Icomm[current_step + i_c].reshape(-1))
                comm_index_one = list(set(comm_index_one_raw))
                comm_index_one.sort(key=comm_index_one_raw.index)
                comm_index_one = [x for x in comm_index_one if self.fact_dict['comm'][x] not in para_aristo]
                comm_text_one = [self.fact_dict['comm'][x] for x in comm_index_one]
                comm_amr_one = [self.amr_dict['comm'][x] for x in comm_text_one]
                para =  comm_text_one
                para_amr = comm_amr_one

                assert len(para) == len(para_amr)

                index_one = [('comm', str(x)) for x in comm_index_one]
                indexs.append(index_one)
                para = "@@".join(para).lower()
                paras.append(para)

                para_amr = "@@".join(para_amr).lower()
                paras_amr.append(para_amr)

            current_step += choice_num
            c_lst = []
            for i in range(len(paras)):
                c_lst.append({"text": choices[i]['text'],
                              "label": choices[i]['label'],
                              'hypo': choices[i]['hypo'],
                              'hypo-amr': choices[i]['hypo-amr'],
                              'para_ori': choices[i]['para'],
                              'para': paras[i],
                              'paras_amr': paras_amr[i],
                              'para_index': indexs[i],
                              })
            save_dict = {
                "id": example["id"],
                "question": {
                    "stem": example["question"]["stem"],
                    "choices": c_lst,
                },
                "answerKey": example["answerKey"],
            }
            qa_annotated.append(save_dict)
        # qa_annotated = read_qa('../Data/begin/iter2-10-10/train-10-10.jsonl')
        qa_post = sort_with_AMR(qa_annotated, max_fact=self.max_fact, num_worker=6)
        return qa_post

    def __len__(self):
        return math.ceil(len(self.qa_f[self.datasetID]) / self.batch_size)
    def __iter__(self):

        processed_data = self.qa_post

        if self.datasetID == "train":
            random.shuffle(processed_data)
            # processed_data = sorted(processed_data, key= lambda x: len(x['question']['choices'][ord(x['answerKey']) - ord('A')]['para_iter']), reverse=False)

        batches = []
        num, data = 0, []
        for x in processed_data:
            num += 1
            data.append(x)
            if num >= self.batch_size:
                batches.append(data)
                num, data = 0, []
        if len(data) != 0:
            batches.append(data)


        for batch in batches:
            processor = processors[self.task_name]()
            examples = processor._create_examples(batch, self.datasetID)
            label_list = processor.get_labels()
            features = convert_examples_to_features(
                examples,
                label_list,
                self.max_len,
                self.model_tokenizer,
                sep_token_extra=True,
                pad_on_left=False,
                pad_token_segment_id=0,
            )
            batch_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
            batch_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
            batch_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
            batch_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

            batch_query_ids, batch_query_mask, batch_fact_vec, iter_mask, iter_segmentation = None, None, None, None, None
            if self.datasetID == "train":
                batch_hypo = []
                batch_chain_global = []
                batch_chain_iter_query = []
                batch_chain_iter_fact = []
                iter_segmentation = []
                for data_one in batch:
                    anskey = normalize(data_one['answerKey'])
                    if anskey == 4:
                        continue
                    choice = data_one['question']['choices'][anskey]
                    hypo_one = choice['hypo']
                    batch_hypo.append(hypo_one)

                    chain_global = [self.vec_dict[x[0]][int(x[1])] for x in choice['para_global']]
                    chain_global = sum(chain_global) / len(chain_global)
                    batch_chain_global.append(chain_global)
                    if len(choice['para_iter']) != 0:
                        sampled_chain = random.sample(choice['para_iter'],k=1)[0]
                        chain_iter = [self.fact_dict[x[0]][int(x[1])] for x in sampled_chain]
                        chain_iter_re = [self.fact_dict[x[0]][int(x[1])] for x in reversed(sampled_chain)]
                        chain_iter_query = [" [SEP] ".join([hypo_one] + chain_iter[:i_i]) for i_i, index_one in enumerate(chain_iter)][1:] \
                                           + [" [SEP] ".join([hypo_one] + chain_iter_re[:i_i]) for i_i, index_one in enumerate(chain_iter_re)][1:]
                        chain_iter_fact = [self.vec_dict[x[0]][int(x[1])] for x in sampled_chain][1:] +\
                                          [self.vec_dict[x[0]][int(x[1])] for x in reversed(sampled_chain)][1:]
                        iter_segmentation.append(len(chain_iter_query))
                        batch_chain_iter_query.extend(chain_iter_query)
                        batch_chain_iter_fact.extend(chain_iter_fact)
                    else:
                        iter_segmentation.append(0)
                iter_segmentation = [sum(iter_segmentation[:i_x + 1]) for i_x, x in enumerate(iter_segmentation)]

                batch_query_ids = []
                batch_query_mask = []
                batch_query = batch_hypo + batch_chain_iter_query
                [(batch_query_ids.extend(x), batch_query_mask.extend(y)) for (x, y) in
                 batchfy(batch_query, self.retriever_tokenizer,
                         pad=self.retriever_tokenizer.pad_token_id, batch_size=len(batch_query))]
                batch_query_ids = torch.tensor(batch_query_ids, dtype=torch.long)
                batch_query_mask = torch.tensor(batch_query_mask, dtype=torch.long)

                batch_fact = batch_chain_global + batch_chain_iter_fact
                if self.fp_16:
                    batch_fact_vec = torch.tensor(batch_fact, dtype=torch.float16)
                else:
                    batch_fact_vec = torch.tensor(batch_fact, dtype=torch.float32)
                if iter_segmentation[-1] == 0:
                    iter_mask = torch.zeros((0, 0), dtype=torch.bool)
                else:
                    iter_mask = torch.zeros((iter_segmentation[-1], iter_segmentation[-1]), dtype=torch.bool)
                    for i, _ in enumerate(iter_segmentation):
                        if i == 0:
                            iter_mask[:iter_segmentation[i], :iter_segmentation[i]] = 1
                        else:
                            iter_mask[iter_segmentation[i - 1]:iter_segmentation[i], iter_segmentation[i - 1]:iter_segmentation[i]] = 1
                    iter_mask.masked_fill_(torch.diag(torch.tensor(iter_segmentation[-1] * [1])), 0)
            yield (batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids,
                   batch_query_ids, batch_query_mask, batch_fact_vec, iter_mask, iter_segmentation)

def prepare_query(query, fact_f, Ilast=None):
    '''
    currently, no beam search
    '''
    if Ilast is None:
        return query
    else:
        data = [query[ix] + ' [SEP] ' + fact_f[y] for ix, x in enumerate(Ilast) for iy, y in enumerate(x)]
        return data

def update_DI(D, I, k):
    if I.shape[1] == k:
        return I
    else:
        num = I.shape[0]
        I_new = I.reshape([num, -1])
        D_new = D.reshape([num, -1])
        Drank = np.argsort(-D_new, axis=1)
        I_out = []
        for i in range(I_new.shape[0]):
            I_out_one = []
            j = 0
            while len(I_out_one) < k and j < k * k:
                if I_new[i][Drank[i][j]] not in I_out_one:
                    I_out_one.append(I_new[i][Drank[i][j]])
                j += 1
            I_out.append(I_out_one)
        return I_out

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def normalize(truth):
    if truth in "ABCDE":
        return ord(truth) - ord("A")
    elif truth in "12345":
        return int(truth) - 1
    else:
        logger.info("truth ERROR! %s", str(truth))
        return None