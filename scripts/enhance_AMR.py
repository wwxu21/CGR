import json
import argparse
def read_openbook(addr):
    with open(addr, 'r') as reader:
        ob = reader.readlines()
        ob = [x.strip() for x in ob]
    return ob

def read_qa(addr):
    qa = []
    with  open(addr, 'r') as reader:
        for line in reader:
            json_line = json.loads(line)
            qa.append(json_line)
    return qa

def read_amr(addr, start=6):
    amrs = []
    ids = []
    id_printed = []
    begin = False
    with open(addr, 'r') as reader:
        for each in reader:
            if each.startswith("# ::id"):
                begin = True
                idx = each.strip().split(' ')[2]
                ids.append(idx)
                amr = []
            if each.strip() == '':
                begin = False
                id_printed.append(idx)
                amrs.append(amr)
            if begin:
                amr.append(each.strip())
    amrs = [amr[start:] for amr in amrs]
    amrs = [''.join(amr) for amr in amrs]
    return amrs


def enhance_amr_arc(amrs, save):

    amrs = read_amr(amrs, start=6)
    with open(save, 'w') as writer:
        for i_line, line in enumerate(amrs):
            amr = amrs[line]
            output_dict = {
                "id": i_line,
                "text": line,
                "amr": amr,
            }
            writer.write(json.dumps(output_dict) + "\n")
def enhance_amr(qa, amrs, save):
    qa = read_qa(qa)
    amrs = read_amr(amrs)
    with open(save, 'w') as writer:
        for i_line, line in enumerate(qa):
            amr4 = amrs[i_line * 4 : i_line * 4 + 4]
            c_lst = []
            for i in range(len(amr4)):
                c_lst.append({"text": line["question"]['choices'][i]['text'],
                              "label": line["question"]['choices'][i]['label'],
                              'hypo': line["question"]['choices'][i]['hypo'],
                              'hypo-amr':amr4[i]})
            output_dict = {
                "id": line["id"],
                "question": {
                    "stem": line["question"]["stem"],
                    "choices": c_lst,
                },
                "fact1": line["fact1"],
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
        train_qa_file = './Data/OpenBook/train.jsonl'
        test_qa_file = './Data/OpenBook/test.jsonl'
        dev_qa_file = './Data/OpenBook/dev.jsonl'

        train_amr_file = './Data/OpenBook/train-amr.txt'
        test_amr_file = './Data/OpenBook/test-amr.txt'
        dev_amr_file = './Data/OpenBook/dev-amr.txt'
        openbook_amr_file = './Data/OpenBook/core-amr.txt'
        arc_amr_file = './Data/OpenBook/comm-amr.txt'

        train_output_file = './Data/begin/obqa/train.jsonl'
        test_output_file = './Data/begin/obqa/test.jsonl'
        dev_output_file = './Data/begin/obqa/dev.jsonl'
        openbook_output_file = './Data/begin/obqa/core.dict'
        arc_output_file = './Data/begin/obqa/obqa.dict'

        enhance_amr(train_qa_file, train_amr_file, train_output_file)
        enhance_amr(dev_qa_file, dev_amr_file, dev_output_file)
        enhance_amr(test_qa_file, test_amr_file, test_output_file)
        enhance_amr_arc(openbook_amr_file, openbook_output_file)
        enhance_amr_arc(arc_amr_file, arc_output_file)