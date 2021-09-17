from tqdm import tqdm
import json
import argparse
def read_qa(addr):
    qa = []
    with  open(addr, 'r') as reader:
        qa_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in qa_tqdm:
            json_line = json.loads(line)
            qa.append(json_line)
    return qa

def read_hypo(addr):
    with open(addr, 'r') as reader:
        hypos = reader.readlines()
    return hypos

def enhance_hypo(qa, hypos, save, corpus='openbook'):
    qa = read_qa(qa)
    hypos = read_hypo(hypos)
    total_choices_num = 0
    with open(save, 'w') as writer:
        for i_line, line in enumerate(qa):
            num_choice = len(line['question']['choices'])
            hypo = hypos[total_choices_num: total_choices_num + num_choice]
            total_choices_num += num_choice
            hypo = [x.strip() for x in hypo]
            c_lst = []
            for i in range(len(hypo)):
                c_lst.append({"text": line["question"]['choices'][i]['text'], "label": line["question"]['choices'][i]['label'], 'hypo': hypo[i]})
            if corpus == "openbook":
                output_dict = {
                    "id": line["id"],
                    "question": {
                        "stem": line["question"]["stem"],
                        "choices": c_lst,
                    },
                    "fact1": line["fact1"],
                    "answerKey": line["answerKey"],
                }
            elif corpus == 'arc':
                output_dict = {
                    "id": line["id"],
                    "question": {
                        "stem": line["question"]["stem"],
                        "choices": c_lst,
                    },
                    "answerKey": line["answerKey"],
                }
            writer.write(json.dumps(output_dict) + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='obqa', type=str, help="which task to perform")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.task == "obqa":
        train_qa_file = './Data/OpenBook/train_complete.jsonl'
        test_qa_file = './Data/OpenBook/test_complete.jsonl'
        dev_qa_file = './Data/OpenBook/dev_complete.jsonl'
        train_hypo_file = './Data/OpenBook/train-hypo.txt'
        test_hypo_file = './Data/OpenBook/test-hypo.txt'
        dev_hypo_file = './Data/OpenBook/dev-hypo.txt'
        train_output_file = './Data/OpenBook/train.jsonl'
        test_output_file = './Data/OpenBook/test.jsonl'
        dev_output_file = './Data/OpenBook/dev.jsonl'
        enhance_hypo(train_qa_file, train_hypo_file, train_output_file)
        enhance_hypo(dev_qa_file, dev_hypo_file, dev_output_file)
        enhance_hypo(test_qa_file, test_hypo_file, test_output_file)
    # train_qa_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-regent-AI2.jsonl'
    # test_qa_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl'
    # dev_qa_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl'
    # train_hypo_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-regent-AI2-hypo.txt'
    # test_hypo_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test-hypo.txt'
    # dev_hypo_file = '..//Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev-hypo.txt'
    # train_output_file = '../Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/train-regent-AI2.jsonl'
    # test_output_file = '../Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/test.jsonl'
    # dev_output_file = '../Data/ARC-14m/ARC-V1-Feb2018-2/ARC-Challenge/dev.jsonl'
    # enhance_hypo(train_qa_file, train_hypo_file, train_output_file, corpus='arc')
    # enhance_hypo(dev_qa_file, dev_hypo_file, dev_output_file, corpus='arc')
    # enhance_hypo(test_qa_file, test_hypo_file, test_output_file, corpus='arc')