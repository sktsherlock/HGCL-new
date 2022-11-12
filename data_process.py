import pandas as pd
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset


def read_data(filename):
    df = pd.read_csv(filename)
    return df

def main():
    # load dgl dataset
    dataset = DglNodePropPredDataset('ogbn-arxiv', root = 'data/ogb/')
    g, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    labels = labels.squeeze().numpy()
    labels.unique().shape[0]
    # make a label.txt 
    f = open('label.txt', 'w')
    split = [None] * len(labels)
    for s, ids in split_idx.items():
        for i in  ids:
            split[i] = s
    
    df = pd.DataFrame({'split': np.array(split), 'label': labels})
    df.to_csv('a.txt', sep='\t', header=False)

    # add train val test in the label.txt file
    # for i in range(len(labels)):
    #     f.write(str(labels[i]) + '\t' + str(split_idx[i]) + '\n')
    # f.close() 
    # for i in range(labels.shape[0]):
    #     if i in split_idx['train']:
    #         f.write('train' + '/t')
    #     elif i in split_idx['valid']:
    #         f.write('val' + '/t')
    #     else:
    #         f.write('test' + '/t')
    #     f.write(str(labels[i]) + '\n')
    # f.close()

if __name__ == '__main__':
    main()