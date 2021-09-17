import argparse
import re
import os
def readf(addr):
    f = []
    with open(addr) as reader:
        for line in reader:
            f.append(line.strip())
    return f

def clean_short(f):
    new_f = []
    for line in f:
        token = line.split()
        if len(token) >= 4:
            new_f.append(line)
    return new_f

def clean_long(f):
    new_f = []
    for line in f:
        token = line.split()
        if len(token) <= 100:
            new_f.append(line)
    return new_f

def clean_url(f):
    new_f = []
    for line in f:
        url_list = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', line)
        if len(url_list) == 0:
            new_f.append(line)
    return new_f

def clean_starts(f):
    new_f = []
    for line in f:
        if line.startswith("_"):
            new_line = line.replace("_","")
        elif line.startswith("-"):
            new_line = line.replace("-", "")
        elif line.startswith("—"):
            new_line = line.replace("—", "")
        elif line.startswith("..."):
            new_line = line.replace("...", "")
        else:
            new_line = line
        new_line = new_line.strip()
        new_f.append(new_line)
    return new_f

def clean_book(f):
    new_f = []
    for line in f:
        book_list = re.findall('pp,?\s?[0-9]+–[0-9]+|:,?\s?[0-9]+–[0-9]+', line)
        if len(book_list) == 0:
            new_f.append(line)
    return new_f

def clean_punc(f):
    new_f = []
    for line in f:
        if line.startswith("#"):
            continue
        elif line.startswith("*"):
            url_list = re.findall('-L\SB-|-R\SB-', line)
            if len(url_list) == 0:
                new_f.append(line.replace("*", "").strip())
        elif line.startswith("•"):
            new_f.append(line.replace("•", "").strip())
        else:
            new_f.append(line)
    return new_f

def clean_quote(f):
    new_f = []
    for line in f:
        url_list = re.findall('&gt|&lt', line)
        if len(url_list) == 0:
            new_f.append(line)
    return new_f

def clean_special(f):
    new_f = []
    for line in f:
        special_list1 = re.findall("[■♂♀©]|R\SB[\s-]+R\SB", line)
        special_list2 = re.findall("@@", line)
        if len(special_list1) == 0 and len(special_list2) == 0:
            new_f.append(line)
    return new_f

def clean_isa(f):
    new_f = []
    for line in f:
        if " isa " in line:
            new_line = line.replace(' isa ', " is a ")
            new_f.append(new_line)
        else:
            new_f.append(line)
    return new_f

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='obqa', type=str, help="which task to perform")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    prefix = './Data/begin/' + args.task
    input_addr = './Data/OpenBook/ARC_Corpus.txt'
    save_addr = './Data/OpenBook/ARC_Clean.txt'
    f = readf(input_addr)
    f = clean_punc(f)
    f = clean_starts(f)
    f = clean_url(f)
    f = clean_book(f)
    f = clean_quote(f)
    f = clean_special(f)
    f = clean_isa(f)
    f = clean_short(f)
    f = clean_long(f)
    f = list(set(f))
    print(len(f))
    with open(save_addr, 'w') as writer:
        for line in f:
            writer.write(line + "\n")


