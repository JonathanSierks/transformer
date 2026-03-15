# Title
little intro

## 1) Context / Key-concepts


## 2) Implementation details

Dataset description

1) Baseline: MLP (toy data)

Task: next character prediction

Architecture:

- Sequence length: 64
- batch size: 16
- embedding dimension: 30
- epochs: 20
- lr: 1e-3
- indput dimension: 64 * 30 = 1920
- hidden dimension: 512
- output dimension: 31 (vocabulary size so 1 class for every character)
- optimizer: standard adam

test loss: 0.774765133857727
test acc: 0.74125

example sentence: "an ( with a systematie dog khile a kawyer . a jy man imp )ie walked ( while a busy dungawbunt ( ran"

2) Transformer (toy data)



3) Transformer (wiki data) tbd


## 3) Results

### 3.1) Baseline MLP (toy data)

| Metrics         | Cora   | PubMed       | CiteSeer     |
|-----------------|---------|------------|----------------|
| test acc (std over 9 runs)           | 79.2 ($\pm0.05$) | 78.39 ($\pm0.5$) | 66.7 ($\pm1.9$)  |
| test acc Kipf & Welling  | 81.5       | 79.0          | 70.3     |
| test acc difference      | 2.3        | 0.61          | 3.3      |
| mean epoch total time    | 0.0015s    | 0.0017        | 0.0011s  |
| mean total train time    | 0.1022     | 0.1900s       |  0.0429  |

plot
![Training curves](results/plots/cora.png) 

short text

### 3.2) Transformer (toy data)

| Metrics         | Cora + full-batch   | Cora + mini-batch | ogbn-arxiv + full-batch| ogbn-arxiv + mini-batch |
|-----------------|---------------------|----------------|--------------------------|-------------------------|
| mean epoch runtime | 0.0015s    | 0.0061s   | 0.0242s | 0.8268s |
| test acc (mean over 3 runs) | 79.31    | 79.18  |  63.85 |  52.8 |

plots
![Training curves](results/plots/ogbn_fb.png)
![Training curves](results/plots/ogbn_fb.png)


text

### 3.3) Transformer (wiki data) tbd

## Project structure

```
repo-name/
├── README.md
├── requirements.txt                 
├── src/
│   ├── model.py                   
│   ├── train.py                
│   ├── data_utils.py              
├── data/
│   └── Planetoid
│   └── ogbn_arxiv                
├── notebooks/
│   └── experiments.ipynb  
├── configs/
│   └── default_config.yaml
│   └── citeseer.yaml
│   └── pubmed.yaml   
│   └── ogbn_arxiv_full_batch.yaml
│   └── pubmed_config_mini_batch.yaml
├── results/
    └── ... 
```

## How to run

Main packages used in this project:
- PyTorch
- NumPy
- wget

Run baseline:

`python -m src.train_baseline`


## References

none