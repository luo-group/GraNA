import argparse
import pickle
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.dataset import EdgeTrainDataset
from src.model import GraNA
from src.utils import direct_compute_deepwalk_matrix, edgeindex2adjacency, laplacian_positional_encoding


def train(learning_rate=0.001, num_negative=1, num_epochs=200, if_orth=True, if_seq=True, batch_size=2**16, species1='hsa', species2='sce', hidden_dim=128, num_layer=3, conv_type='GEN', seed=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading data ...')

    '''loading integrated graph'''
    nx_g12 = nx.read_edgelist('data/physical_interaction/'+species1+'_'+species2+'.edgelist')
    edge_index = np.vstack([np.array(e, dtype=int) for e in nx_g12.edges] + [np.flip(np.array(e, dtype=int)) for e in nx_g12.edges])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    tg = nx.relabel_nodes(nx_g12, lambda x: int(x))
    g = nx.Graph()
    g.add_nodes_from(sorted(tg.nodes(data=True)))
    g.add_edges_from(tg.edges(data=True))


    '''laplacian positional embedding'''
    lap_pos_enc = laplacian_positional_encoding(g, hidden_dim).to(device)

    '''netmf calculation'''
    try:
        normalized_matrix = pickle.load(open('data/'+species1+'_'+species2+'normalized_matrix.pkl', 'rb'))
    except:
        window = 10
        adjacency_matrix = nx.adjacency_matrix(g)
        dw_matrix = direct_compute_deepwalk_matrix(
            adjacency_matrix, window).toarray()

        norms = np.linalg.norm(dw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = dw_matrix / norms

        adjacency_matrix = torch.from_numpy(
            adjacency_matrix.toarray()).float().to(device)
        dw_matrix = torch.from_numpy(dw_matrix).float()
        normalized_matrix = torch.from_numpy(
            normalized_matrix).float()

        pickle.dump(normalized_matrix, open('data/'+species1+'_'+species2+'normalized_matrix.pkl', 'wb'))
    x = normalized_matrix
    print(x.shape)

    '''load ortholog as anchor links'''
    nx_g_orth = nx.read_edgelist('data/ortholog/'+species1+'_'+species2+'_relabeled.edgelist')
    edge_index12 = np.vstack([np.array(e, dtype=int) for e in nx_g_orth.edges] + [np.flip(np.array(e, dtype=int)) for e in nx_g_orth.edges])
    edge_index12 = torch.tensor(edge_index12, dtype=torch.long).t().contiguous()
    print('Number of orthologs:',edge_index12.shape[1]/2)

    '''load sequence similarity as anchor links'''
    nx_g_sim = nx.read_edgelist('data/sequence/'+species1+'_'+species2+'_relabeled.edgelist')
    print(len(nx_g_sim.edges))
    nx_g_onto = nx.read_edgelist('data/sce_spo/'+species1+'_'+species2+'_relabeled.edgelist')
    nx_g_sim.remove_edges_from(nx_g_onto.edges)
    print(len(nx_g_sim.edges))
    edge_index12_sim = np.vstack([np.array(e, dtype=int) for e in nx_g_sim.edges] + [np.flip(np.array(e, dtype=int)) for e in nx_g_sim.edges])
    edge_index12_sim = torch.tensor(edge_index12_sim, dtype=torch.long).t().contiguous()
    print('Number of sequence similarities:', edge_index12_sim.shape[1]/2)


    '''load data splits'''
    split_number, total_number, train_mask, valid_mask, test_mask = pickle.load(open('data/split/'+species1+'_'+species2+'_seed'+str(seed)+'.pkl', 'rb'))
    train_mask1 = set([n for n in train_mask if n < split_number])
    train_mask2 = set(train_mask) - set(train_mask1)
    valid_mask1 = list(set([n for n in valid_mask if n < split_number]))
    valid_mask2 = list(set(valid_mask) - set(valid_mask1))
    test_mask1 = list(set([n for n in test_mask if n < split_number]))
    test_mask2 = list(set(test_mask) - set(test_mask1))

    '''load ontology for train/valid/test'''
    nx_g_onto = nx.read_edgelist('data/sce_spo/'+species1+'_'+species2+'_relabeled.edgelist')
    print(len(nx_g_onto.edges))
    nx_g_onto.remove_nodes_from(set(nx_g_onto.nodes).difference([str(n) for n in train_mask]))
    print(len(nx_g_onto.edges))
    train_edge_index_onto = np.vstack([np.array(e,dtype=int) for e in nx_g_onto.edges] 
                                    + [np.flip(np.array(e,dtype=int)) for e in nx_g_onto.edges])
    train_edge_index_onto = train_edge_index_onto[train_edge_index_onto[:,0]<train_edge_index_onto[:,1]]
    train_edge_index_onto = torch.tensor(train_edge_index_onto, dtype=torch.long).t().contiguous().to(device)

    nx_g_onto = nx.read_edgelist('data/sce_spo/'+species1+'_'+species2+'_relabeled.edgelist')
    nx_g_onto.remove_nodes_from(set(nx_g_onto.nodes).difference([str(n) for n in valid_mask]))
    print(len(nx_g_onto.edges))
    valid_edge_index_onto = np.vstack([np.array(e, dtype=int) for e in nx_g_onto.edges] 
                                    + [np.flip(np.array(e, dtype=int)) for e in nx_g_onto.edges])
    valid_target_matrix = edgeindex2adjacency(valid_mask1, valid_mask2, valid_edge_index_onto, edge_index12, total_number)

    nx_g_onto = nx.read_edgelist('data/sce_spo/'+species1+'_'+species2+'_relabeled.edgelist')
    nx_g_onto.remove_nodes_from(set(nx_g_onto.nodes).difference([str(n) for n in test_mask]))
    print(len(nx_g_onto.edges))
    test_edge_index_onto = np.vstack([np.array(e, dtype=int) for e in nx_g_onto.edges] 
                                    + [np.flip(np.array(e, dtype=int)) for e in nx_g_onto.edges])
    test_target_matrix = edgeindex2adjacency(test_mask1, test_mask2, test_edge_index_onto, edge_index12, total_number)

    ontology1_eval = np.sum(valid_target_matrix, axis=1)
    ontology_eval_mask1 = np.where(ontology1_eval > 0)[0].tolist()
    ontology2_eval = np.sum(valid_target_matrix, axis=0)
    ontology_eval_mask2 = np.where(ontology2_eval > 0)[0].tolist()

    ontology1_test = np.sum(test_target_matrix, axis=1)
    ontology_test_mask1 = np.where(ontology1_test > 0)[0].tolist()
    ontology2_test = np.sum(test_target_matrix, axis=0)
    ontology_test_mask2 = np.where(ontology2_test > 0)[0].tolist()

    valid_target_matrix_prc = torch.tensor(valid_target_matrix[ontology_eval_mask1][:,ontology_eval_mask2], dtype=torch.float).to(device).flatten()
    valid_target_matrix_roc = torch.tensor(valid_target_matrix[ontology_eval_mask1][:,ontology_eval_mask2], dtype=torch.int).to(device).flatten()
    test_target_matrix_prc = torch.tensor(test_target_matrix[ontology_test_mask1][:,ontology_test_mask2], dtype=torch.float).to(device).flatten()
    test_target_matrix_roc = torch.tensor(test_target_matrix[ontology_test_mask1][:,ontology_test_mask2], dtype=torch.int).to(device).flatten()


    print('Building heterogeneous data...')
    data = HeteroData()
    data['protein'].x = x
    data['protein', 'interact', 'protein'].edge_index = edge_index
    if if_orth:
        data['protein', 'ortholog', 'protein'].edge_index = edge_index12
    if if_seq:
        data['protein', 'sequence', 'protein'].edge_index = edge_index12_sim
    data = data.to(device)

    print('Building model ...')
    model = GraNA(data.metadata(), g.number_of_nodes(), split_number, hidden_dim=hidden_dim, num_layer=num_layer, conv_type=conv_type)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss_fn1 = nn.BCELoss()

    st = time.time()
    train_dataset = EdgeTrainDataset(train_edge_index_onto, device, split_number, total_number, num_negative)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    ed = time.time()
    print('Dataset building time:', ed-st)

    metric_auprc = torchmetrics.AveragePrecision()
    metric_auprc.to(device)
    metric_auroc = torchmetrics.AUROC()
    metric_auroc.to(device)

    print('Training ...')

    best_auprc = 0
    best_test_auroc, best_test_auprc = 0, 0


    for epoch in range(num_epochs):

        model.train()

        st = time.time()
        train_dataset.shuffle()
        ed = time.time()
        print('Dataset shuffling time:', ed-st)

        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            out = model(data.x_dict, data.edge_index_dict, lap_pos_enc)['protein']
            z = torch.cat([out[batch['first_col']],out[batch['second_col']]], dim=1)
            loss = loss_fn1(model.decode(z), batch['label'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        
        loss = total_loss / len(dataloader)

        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict, lap_pos_enc)['protein']

            z = torch.cat([out[valid_mask1, :][ontology_eval_mask1].repeat(1, len(ontology_eval_mask2)).reshape(-1, hidden_dim), out[valid_mask2][ontology_eval_mask2].repeat(len(ontology_eval_mask1),1)], dim=1)
            predicted_score_valid = model.decode(z)
            predicted_score_valid_matrix = predicted_score_valid.squeeze(1)

            valid_auprc = metric_auprc(predicted_score_valid_matrix, valid_target_matrix_prc).item()
            valid_auroc = metric_auroc(predicted_score_valid_matrix, valid_target_matrix_roc).item()

            z = torch.cat([out[test_mask1, :][ontology_test_mask1].repeat(1, len(ontology_test_mask2)).reshape(-1, hidden_dim), out[test_mask2][ontology_test_mask2].repeat(len(ontology_test_mask1),1)], dim=1)
            predicted_score_test = model.decode(z)
            predicted_score_test_matrix = predicted_score_test.squeeze(1)

            test_auprc = metric_auprc(predicted_score_test_matrix, test_target_matrix_prc).item()
            test_auroc = metric_auroc(predicted_score_test_matrix, test_target_matrix_roc).item()
            
            if valid_auprc > best_auprc:
                best_auprc = valid_auprc
                best_valid_auroc = valid_auroc
                best_test_auprc = test_auprc
                best_test_auroc = test_auroc

                if if_orth and if_seq:
                    torch.save(model.state_dict(), 'results/model/'+species1+'_'+species2+'_'+str(seed)+'.pt')
                elif if_orth and not if_seq:
                    torch.save(model.state_dict(), 'results/model/'+species1+'_'+species2+'_het_orth_'+str(seed)+'.pt')
                elif not if_orth and if_seq:
                    torch.save(model.state_dict(), 'results/model/'+species1+'_'+species2+'_het_sim_'+'_'+str(seed)+'.pt')
                elif not if_orth and not if_seq:
                    torch.save(model.state_dict(), 'results/model/'+species1+'_'+species2+'_het_no_'+str(seed)+'.pt')

        print(f'epoch:{epoch}, loss:{loss:.4f}, best test auprc:{best_test_auprc:.4f}, best test auroc:{best_test_auroc:.4f}, valid auroc:{valid_auroc:.4f}, valid auprc:{valid_auprc:.4f}, test auroc:{test_auroc:.4f}, test auprc:{test_auprc:.4f}')


def main(args):
    train(species1=args.first, 
          species2=args.second, 
          num_epochs=args.num_epochs,
          batch_size=2**args.batch_size,
          learning_rate=args.learning_rate,
          hidden_dim=args.hidden_dim, 
          num_layer=args.num_layer, 
          conv_type=args.conv_type,
          seed=args.seed,
          if_orth=args.orth,
          if_seq=args.seq)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--first', '-p1', default='sce', type=str, help='first ppi network')
    parser.add_argument('--second', '-p2', default='spo', type=str, help='second ppi network')
    parser.add_argument('--num_epochs', '-n', default=200, type=int, help='num epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='log2 batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_dim', '-d', default=128, type=int, help='hidden dimension')
    parser.add_argument('--num_layer', '-l', default=7, type=int, help='num layers')
    parser.add_argument('--conv_type', '-c', default='GEN', type=str, help='convolution layer')
    parser.add_argument('--seed', '-s', default=0, type=int, help='Seed for data split')
    parser.add_argument('--orth', action='store_false', help='Use orthologs as anchor links')
    parser.add_argument('--seq', action='store_false', help='Use sequence similarity as anchor links')
    args = parser.parse_args()
    main(args)
