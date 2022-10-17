import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class EdgeTrainDataset(Dataset):
    def __init__(self, edge_index, device, split_number, total_number, num_negatives=1):

        self.positives = []
        self.negatives = []
        self.positives_dict1, self.positives_dict2 = {}, {}

        self.split_number = split_number
        self.total_number = total_number

        self.device = device

        first_col, second_col, gt = [], [], []
        species1, species2 = edge_index.cpu()
        for p1_, p2_ in tqdm(zip(species1, species2), desc='loading positive edges'):
            p1, p2 = int(p1_), int(p2_)

            self.positives.append((p1,p2))

            self.positives_dict1.setdefault(p1, set([]))
            self.positives_dict1[p1].add(p2)
            self.positives_dict2.setdefault(p2, set([]))
            self.positives_dict2[p2].add(p1)
        
        self.label = torch.unsqueeze(torch.tensor([1]*len(self.positives)+[0]*len(self.positives), dtype=torch.float).to(self.device),1)

    def shuffle(self, noise_ratio=0):
        
        num_sample_negatives = int(5 * len(self.positives))
        negatives_first_col = random.choices(list(self.positives_dict1.keys()),k=num_sample_negatives)
        negatives_second_col = random.choices(list(self.positives_dict2.keys()),k=num_sample_negatives)

        self.negatives = []
        cnt = 0
        num_negatives = len(self.positives)
        while len(self.negatives) < num_negatives:
            if negatives_second_col[cnt] not in self.positives_dict1[negatives_first_col[cnt]]:
                self.negatives.append([negatives_first_col[cnt], negatives_second_col[cnt]])
            cnt += 1


        self.first_col, self.second_col = np.array(self.positives+self.negatives).transpose().tolist()

    def __len__(self):
        return 2 * len(self.positives)
    
    def __getitem__(self, index):
        return {'first_col':self.first_col[index], 'second_col':self.second_col[index], 'label':self.label[index]}
