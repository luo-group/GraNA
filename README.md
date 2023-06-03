# GraNA
This is the official repository for the paper *Supervised biological network alignment with graph neural networks* [[link](https://www.biorxiv.org/content/10.1101/2023.04.24.538184v1)]. **GraNA** is a deep learning framework for supervised biological network alignment (NA). Employing graph neural networks (GNNs), GraNA utilizes within-network interactions and across-network anchor links for learning protein representations and predicting functional correspondence between across-species proteins.


## Install dependencies
```
conda create -n grana python=3.9
conda activate grana
```
We use pytorch 1.12.1 and pytorch-geometrics 2.1.0.post1, which can be installed with the proper version for your cuda following the instructions on their offical website.
```
pip install -r requirements.txt
conda install scipy mkl-service
```


## Download data
In our paper, we use the datasets provided by the authors of ETNA for benchmarking. An example data can be downloaded from https://github.com/ylaboratory/ETNA.


## Directories
Once the data from the above url are downloaded, the file structure can be formulated as follows:
```
.
|-- data
|   |-- emb
|   |-- ortholog
|   |   |-- sce_spo_orthomcl.txt
|   |-- physical_interaction
|   |   |-- sce_physical_pairs.txt
|   |   |-- spo_physical_pairs.txt
|   |--sce_spo
|   |   |--sce_spo_ontology_pairs_expert.txt
|   |--sequence
|   |   |--sce_spo_relabeled.edgelist
|   |--split
|-- code
|   |-- dataset.py
|   |-- load_data.py
|   |-- model.py
|   |-- utils.py
|-- results
|   |--model
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- train.py
```

## Load data
To preprocess the data loaded, run the following:
```
python src/load_data.py
```

## Train GraNA
To train GraNA after loading the data, run the following:
```
python train.py
```