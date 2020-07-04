# SelfTask-GNN
A PyTorch implementation of "Self-supervised Learning on Graphs: Deep Insights and New Directions". [[paper]](https://arxiv.org/abs/2006.10141)  

In this paper, we first deepen our understandings on when, why, and which strategies of SSL work with GNNs by empirically studying numerous basic SSL pretext tasks on graphs. Inspired by deep insights from the empirical studies, we propose a new direction *SelfTask* to build advanced pretext tasks that are able to achieve state-of-the-art performance on various real-world datasets.

## Requirements
See that in https://github.com/ChandlerBang/SelfTask-GNN/blob/master/requirements.txt

## Run our code
Clone the repository
```
git clone https://github.com/ChandlerBang/SelfTask-GNN.git
cd SelfTask-GNN
pip install -r requirements.txt
```

To reproduce the performance reported in the paper, you can run the bash files in folder `scripts`. 
```
sh scripts/selftask/cora_CorrectedLabel_ICA.sh
sh scripts/selftask/cora_CorrectedLabel_LP.sh
```


## Acknowledgement
This repository is modified from DropEdge [(https://github.com/DropEdge/DropEdge)](https://github.com/DropEdge/DropEdge). We sincerely thank them for their contributions.

## Cite
For more information, you can take a look at the [paper](https://arxiv.org/abs/2006.10141) 

If you find this repo to be useful, please cite our paper. Thank you.
```
@misc{jin2020selfsupervised,
    title={Self-supervised Learning on Graphs: Deep Insights and New Direction},
    author={Wei Jin and Tyler Derr and Haochen Liu and Yiqi Wang and Suhang Wang and Zitao Liu and Jiliang Tang},
    year={2020},
    eprint={2006.10141},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

