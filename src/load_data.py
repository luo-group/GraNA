import argparse
import os
import pickle
import random
from collections import defaultdict

import networkx as nx


def keep_distinct_node(g):
    '''
    Keep nodes with distinct neighborhood in the graph
    https://github.com/ylaboratory/ETNA/blob/master/src/load_data.py
    '''
    A = nx.adjacency_matrix(g).toarray()
    a_dict = defaultdict(set)
    node_list = list(g.nodes())
    for i in range(A.shape[0]):
        a_dict[tuple(A[i])].add(node_list[i])
    distinct_nodes = set()
    for s in a_dict.values():
        if len(s) < 2:
            distinct_nodes.add(list(s)[0])
    removed_nodes = set(g.nodes()).difference(distinct_nodes)
    g.remove_nodes_from(removed_nodes)


def map_nodes(g):
    '''
    map from node name to node index & node index to node name
    '''
    map_n2i, map_i2n = dict({}), dict({})
    for i,j in enumerate(g.nodes()):
        map_n2i[j] = i
        map_i2n[i] = j
    return map_n2i, map_i2n


def load_ppi_graph(file):
    '''
    load and preprocess ppi graph
    modified from https://github.com/ylaboratory/ETNA/blob/master/src/load_data.py
    '''

    nx_g = nx.read_edgelist(file, create_using=nx.DiGraph())
    for edge in nx_g.edges():
        nx_g[edge[0]][edge[1]]['weight'] = 1 # weight for node2vec
    print(f'Read as directed: {len(nx_g)} nodes, {nx_g.size()} edges')
    
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    nx_g.remove_nodes_from(list(nx.isolates(nx_g)))
    print(f'Self loop and isolates removed: {len(nx_g)} nodes, {nx_g.size()} edges')

    nx_g = nx_g.to_undirected()
    print(f'Convert to undirected: {len(nx_g)} nodes, {nx_g.size()} edges')

    node_len = 0
    while node_len != len(nx_g.nodes()):
        node_len = len(nx_g.nodes())
        keep_distinct_node(nx_g)
        nx_g.remove_nodes_from(list(nx.isolates(nx_g)))
    print(f'Keep distinct nodes: {len(nx_g)} nodes, {nx_g.size()} edges')

    return nx_g


def load_both_ppi(species1, species2, parent_dir='data'):
    '''
    load data for GraNA
    '''

    f_ppi1 = os.path.join(parent_dir, 'physical_interaction', species1+'_physical_pairs.txt')
    f_ppi2 = os.path.join(parent_dir, 'physical_interaction', species2+'_physical_pairs.txt')

    nx_g1 = load_ppi_graph(f_ppi1)
    nx_g2 = load_ppi_graph(f_ppi2)
    
    nx_g12 = nx.union(nx_g1, nx_g2, rename=(species1, species2))
    map_n2i, map_i2n = map_nodes(nx_g12)

    nx_g12_relabeled = nx.relabel_nodes(nx_g12, map_n2i)

    print(f'Integrated graph: {len(nx_g12_relabeled.nodes())} nodes, {nx_g12_relabeled.size()} edges')

    # save edgelist of the integrated graph
    nx.write_edgelist(nx_g12_relabeled, os.path.join(parent_dir, 'physical_interaction/'+species1+'_'+species2+'.edgelist'))

    # nx.write_adjlist(nx_g12_relabeled, os.path.join(parent_dir, 'physical_interaction/'+species1+'_'+species2+'.adjlist'))

    pickle.dump(map_n2i, open(os.path.join(parent_dir, 'emb/'+species1+'_'+species2+'map_n2i.pkl'), 'wb'))
    pickle.dump(map_i2n, open(os.path.join(parent_dir, 'emb/'+species1+'_'+species2+'map_i2n.pkl'), 'wb'))

    return nx_g1, nx_g2, nx_g12_relabeled


def load_orthologs(species1='sce', species2='spo', parent_dir='data'):
    '''
    load ortholog files
    modified from https://github.com/ylaboratory/ETNA/blob/master/src/load_data.py
    '''

    orth_file = os.path.join(parent_dir, 'ortholog/'+species1+'_'+species2+'_orthomcl.txt')
    with open(orth_file, 'r') as f:
        data = f.readlines()[1:]
    
    new_file = os.path.join(parent_dir,'ortholog/'+species1+'_'+species2+'.edgelist')
    with open(new_file, 'w') as f:
        for d in data:
            p1, p2, weight = d.strip().split()
            f.write(' '.join([species1+p1, species2+p2, '{\'weight\': '+weight+'}'])+'\n')

    nx_g_orth = nx.read_edgelist(os.path.join(parent_dir, 'ortholog/'+species1+'_'+species2+'.edgelist'))

    map_n2i = pickle.load(open(os.path.join(parent_dir, 'emb/'+species1+'_'+species2+'map_n2i.pkl'), 'rb'))

    removed_nodes = set(nx_g_orth.nodes()).difference(map_n2i.keys())
    nx_g_orth.remove_nodes_from(removed_nodes)

    nx_g_orth = nx.relabel_nodes(nx_g_orth, map_n2i)

    nx.write_edgelist(nx_g_orth, os.path.join(parent_dir, 'ortholog/'+species1+'_'+species2+'_relabeled.edgelist'))

    return nx_g_orth


def load_ontology(species1='sce', species2='spo', parent_dir='data'):
    '''
    load ontology file
    modified from https://github.com/ylaboratory/ETNA/blob/master/src/load_data.py
    '''
    onto_file = os.path.join(parent_dir, 'sce_spo/'+species1+'_'+species2+'_ontology_pairs_expert.txt')
    with open(onto_file, 'r') as f:
        data = f.readlines()

    new_file = os.path.join(parent_dir, 'sce_spo/'+species1+'_'+species2+'_ontology.edgelist')
    with open(new_file, 'w') as f:
        for d in data:
            p1, p2 = d.strip().split()
            f.write(' '.join([species1+p1, species2+p2])+'\n')
    
    nx_g_onto = nx.read_edgelist(os.path.join(parent_dir, 'sce_spo/'+species1+'_'+species2+'_ontology.edgelist'))

    map_n2i = pickle.load(open(os.path.join(parent_dir, 'emb/'+species1+'_'+species2+'map_n2i.pkl'), 'rb'))

    removed_nodes = set(nx_g_onto.nodes()).difference(map_n2i.keys())
    nx_g_onto.remove_nodes_from(removed_nodes)

    nx_g_onto = nx.relabel_nodes(nx_g_onto, map_n2i)

    nx.write_edgelist(nx_g_onto, os.path.join(parent_dir, 'sce_spo/'+species1+'_'+species2+'_relabeled.edgelist'))


def split_data_final(species1, species2, nx_g1, nx_g2, g12, nx_g_orth, seed, ratio=0.7, parent_dir='data'):
    '''
    split data for training and testing
    '''

    total_number = len(g12.nodes)
    split_number = len(nx_g1.nodes)
    
    map_n2i = pickle.load(open(os.path.join(parent_dir, 'emb/'+species1+'_'+species2+'map_n2i.pkl'), 'rb'))

    num_training_1 = int(ratio * len(nx_g1.nodes))
    num_training_2 = int(ratio * len(nx_g2.nodes))

    valid_ratio = 0.8 - ratio
    num_valid_1 = int(valid_ratio * len(nx_g1.nodes))
    num_valid_2 = int(valid_ratio * len(nx_g2.nodes))

    # sample training samples
    random.seed(seed)
    train_nodes_g1 = random.sample(range(split_number), num_training_1)
    random.seed(seed)
    train_nodes_g2 = random.sample(range(split_number, total_number), num_training_2)

    # sample valid samples
    random.seed(seed)
    valid_nodes_g1 = random.sample(set(range(split_number)).difference(set(train_nodes_g1)), num_valid_1)
    random.seed(seed)
    valid_nodes_g2 = random.sample(set(range(split_number, total_number)).difference(set(train_nodes_g2)), num_valid_2)

    # acquire test samples
    test_nodes_g1 = list(set(range(split_number)).difference(train_nodes_g1).difference(valid_nodes_g1))
    test_nodes_g2 = list(set(range(split_number, total_number)).difference(train_nodes_g2).difference(valid_nodes_g2))

    train_nodes = list(sorted([n for n in train_nodes_g1 + train_nodes_g2]))
    valid_nodes = list(sorted([n for n in valid_nodes_g1 + valid_nodes_g2]))
    test_nodes = list(sorted([n for n in test_nodes_g1 + test_nodes_g2]))

    assert len(train_nodes) + len(valid_nodes) + len(test_nodes) == total_number
    assert len(train_nodes_g1) + len(valid_nodes_g1) + len(test_nodes_g1) == split_number

    train_mask, valid_mask, test_mask = train_nodes, valid_nodes, test_nodes

    print(len(train_mask), len(valid_mask), len(test_mask))

    pickle.dump([split_number, total_number, train_mask, valid_mask, test_mask], open(os.path.join(parent_dir, 'split/'+species1+'_'+species2+'_seed'+str(seed)+'.pkl'), 'wb'))


def preprocess(species1='sce', species2='spo'):

    print('Loading both PPI...')
    g1, g2, g12 = load_both_ppi(species1, species2)

    print('Loading orthologs...')
    g_orth = load_orthologs(species1, species2)

    print('Loading ontology...')
    load_ontology(species1, species2)

    print('Splitting data...')
    # split_data(species1, species2, total_number)
    for seed in range(5):
        split_data_final(species1, species2, g1, g2, g12, g_orth, seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--first', '-p1', default='sce', type=str, help='first ppi network')
    parser.add_argument('--second', '-p2', default='spo', type=str, help='second ppi network')
    args = parser.parse_args()
    preprocess(args.first, args.second)
