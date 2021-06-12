# Accurate Graph-Based PU Learning without Class Prior

## Abstract
How can we classify graph-structured data only with positive labels? Graph-based positive-unlabeled (PU) learning is to train a binary classifier given only the positive labels when the relation between examples is given as a graph. The problem is of great importance for various real-world scenarios such as detecting malicious users in a social network, which are difficult to be modeled by supervised learning when the true negative labels are absent. However, previous works for graph-based PU learning assume that the prior distribution of positive examples is known in advance, which is not true in many real-world cases. In this work, we propose GRAB (Graph-based Risk minimization with iterAtive Belief propagation), a novel end-to-end approach for graph-based PU learning that requires no class prior. GRAB models a given graph as a Markov network and iteratively runs the marginalization step that estimates the unknown priors of target variables, and the update step that trains a classifier network utilizing the computed priors in the objective function. Extensive experiments on five datasets show that GRAB provides the state-of-the-art performance, even compared with previous methods that are given the true prior.

## Code Description
- `src/models/gnn.py` contains the architecture of a GCN classifier.
- `src/models/pgm.py` contains the graphical inference model to run loopy belief propagation over the given graph.
- `src/data.py` contains functions to get and preprocess graph datasets.
- `src/main.py` trains the model.

## Data Overview
| **Dataset**      |                            **Path or Package**                       | 
|:--------------:    |                          :----------:                      | 
|   **Cora**         |   `torch_geometric.datasets.Planetoid`   | 
| **Citeseer**     |    `torch_geometric.datasets.Planetoid`  | 
| **Cora-ML**     | `torch_goemetric.datasets.CitationFull`     | 
| **WikiCS**     | `torch_goemetric.datasets.wikics.WikiCS`     | 
| **MMORPG**     | `data/raw/mmorpg`  |  

* We load four datasets (Cora, Citeseer, Cora_ML, WiKiCS) from the Torch Geometric package.
    In our version, we modified the datasets to have binary classes (PN) with the major class as positive and the other classes as negative.
* We also included a private MMORPG dataset, whose information is summerized as below:
    - Number of nodes: 6,312
    - Number of features: 136
    - Number of edges: 68,012
    - Number of positive nodes: 298
    - Number of unknown nodes: 5,912
    - Number of true negative: 401

### Dependencies
- CUDA 10.0
- python 3.6.8
- torch==1.4.0
- torch-geometric==1.6.3
- numpy==1.19.4
- pandas==1.1.4
- scikit-learn==0.23.2
- scipy==1.5.3
- wheel==0.36.2

### Simple Demo
You can run the demo sript by `bash run_code.sh`.
It trains GRAB on Cora, Citeseer, Cora-ML, WikiCS, and MMORPG, and saves the trained model.
Then, it evaluates the trained model in terms of the F1 score and accuracy. 
- `--data`: cora, citeseer, cora-ml, wikics, mmorpg.
- `--seed`: random seed {0-9}.

#### Results of the Demo
| **Dataset**      |   **F1 score** |   **Accuracy** | 
|:--------------:    |:------:    |:------:    |
| **Cora**    | 80.4     | 93.0     |
| **Citeseer**   | 69.7     | 92.9     |
| **Cora-ML**         | 85.0     | 94.9     |
| **WiKiCS**         | 78.8     | 93.8     |
| **MMORPG**         | 94.2     | 97.0     |

#### Used Hyperparameters 
We briefly summarize the hyperparameters.

* Hyperparameters of GRAB
    - `data`: name of the dataset.
    - `gpu`: index of gpu to use
    - `seed`: random seed (any integer)
    - `epoch`: number of epochs to train the model.
    - `patience`: number of patience to train the model.
    - `trn-ratio`: ratio of positive nodes among the all positive ones for training.
    - `layer`: number of GCN layers for the model.
    - `units`: number of GCN units for the model.
    - `mu-iters`: number of maximum iteration of marginalization-update step for the model.
    - `potential`: potential value for loopy beilef propagation.

#### Detailed Usage
You can reproduce results with the following command:
```shell
python main.py --seed 0 --data cora
python main.py --seed 0 --data citeseer
python main.py --seed 0 --data cora-ml
python main.py --seed 0 --data wikics --epoch 2000 --patience 2000
python main.py --seed 0 --data mmorpg
```

