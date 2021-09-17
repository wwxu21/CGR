# CGR

Code for our **EMNLP Findings 2021** paper,

**Exploiting Reasoning Chains for Multi-hop Science Question Answering**

Weiwen Xu, Yang Deng, Huihui Zhang, Deng Cai and Wai Lam.

## Data Preparation
We present the results on [OpenBookQA](https://leaderboard.allenai.org/open_book_qa/submissions/get-started) and [ARC-Challenge](https://allenai.org/data/arc) in our paper. Due to the license issue, please directly download the datasets from their corresponding websites.

## Data Annotation
We use [this repo](https://github.com/kelvinguu/qanli) as our hypothesis generator and [AMR-gs](https://github.com/jcyk/AMR-gs) as our AMR parser. Please follow their instructions to annotate hypothesis and AMR for the datasets respectively.

Once annotated, please organize the annotated files in the following directory (e.g. OpenBookQA)

    - Data/
        - OpenBook/
            - train-complete.jsonl (train/dev/test original datasets)
            - dev-complete.jsonl
            - test-complete.jsonl
            - openbook.txt
            - ARC_Corpus.txt
            - train-hypo.txt (train/dev/test hypotheses)
            - dev-hypo.txt
            - test-hypo.txt
            - train-amr.txt (train/dev/test AMRs)
            - dev-amr.txt
            - test-amr.txt
            - core-amr.txt (core fact AMRs from open-book)
            - comm-amr.txt (common fact AMRs from ARC-Corpus)
Please use `scripts/clean_corpus.py` to clean the ARC-Corpus to remove noisy sentences.
## Preprocessing
1. Add hypothesis to the original datasets:

`bash enhance_hypo.sh`

2. Add AMR to the hypothesis-enhanced datasets as well as cache all facts AMR:

`bash enhance_AMR.sh`

3. Cache all dense vectors for evidence facts:

`bash cache_vector.sh`

Once get all preprocessed, you will get the following directory:

    - Data/
        - begin/
	    - obqa/
            	- train.jsonl
            	- dev.jsonl
            	- test.jsonl
            	- core.dict (amr file)
                - core.npy (vector file)
                - obqa.dict
                - obqa.npy
## Training
`bash finetune.sh`

## Citation
If you find this work useful, please star this repo and cite our paper as follows:
```
@article{xu2021exploiting,
  title={Exploiting Reasoning Chains for Multi-hop Science Question Answering},
  author={Xu, Weiwen and Deng, Yang and Zhang, Huihui and Cai, Deng and Lam, Wai},
  journal={arXiv preprint arXiv:2109.02905},
  year={2021}
}
```
