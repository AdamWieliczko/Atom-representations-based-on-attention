from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.utils import shuffle
from representations import defaultRepresentation, representation1, representation10, representationAll
from rdkit.Chem import PandasTools


class Featurizer:
    def __init__(self, y_column, smiles_col='Drug', **kwargs):
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.__dict__.update(kwargs)
    
    def __call__(self, df):
        raise NotImplementedError()

class GraphFeaturizer(Featurizer):
    def __call__(self, df, getRepresentation):
        graphs = []
        labels = []
        for i, row in df.iterrows():
            y = row[self.y_column]
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            
            edges = []
            for bond in mol.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edges.append((begin, end))  # TODO: Add edges in both directions
            edges = np.array(edges)
            
            nodes = []
            for atom in mol.GetAtoms():
                # print(atom.GetAtomicNum(), atom.GetNumImplicitHs(), atom.GetTotalNumHs(), atom.GetSymbol(), atom.GetNumExplicitHs(), atom.GetTotalValence())
                results = getRepresentation(atom)
                # print(results)
                nodes.append(results)
            nodes = np.array(nodes)
            
            graphs.append((nodes, edges.T))
            labels.append(y)
        labels = np.array(labels)
        return [Data(
            x=torch.FloatTensor(x), 
            edge_index=torch.LongTensor(edge_index), 
            y=torch.FloatTensor([y])
        ) for ((x, edge_index), y) in zip(graphs, labels)]


class ECFPFeaturizer(Featurizer):
    def __init__(self, y_column, radius=2, length=1024, **kwargs):
        self.radius = radius
        self.length = length
        super().__init__(y_column, **kwargs)
    
    def __call__(self, df):
        fingerprints = []
        labels = []
        for i, row in df.iterrows():
            y = row[self.y_column]
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.length)
            fingerprints.append(fp)
            labels.append(y)
        fingerprints = np.array(fingerprints)
        labels = np.array(labels)
        return fingerprints, labels
    



def GetDataCSV(path, y, smiles):
    dataset = pd.read_csv(path)


    dataset = shuffle(dataset)

    train_dataset = dataset[:int(len(dataset) * 0.7)]
    val_dataset = dataset[int(len(dataset) * 0.7): int(len(dataset) * 0.85)]
    test_dataset = dataset[int(len(dataset) * 0.85):]
    

    featurizer = ECFPFeaturizer(y_column=y, smiles_col=smiles)
    X_train, y_train = featurizer(train_dataset)
    X_valid, y_valid = featurizer(val_dataset)
    X_test, y_test = featurizer(test_dataset)

    batch_size = 64

    featurizer = GraphFeaturizer(y, smiles_col=smiles)

    train_loader1 = GraphDataLoader(featurizer(train_dataset, representation1), batch_size=batch_size, shuffle=True)
    valid_loader1 = GraphDataLoader(featurizer(val_dataset, representation1), batch_size=batch_size)
    test_loader1 = GraphDataLoader(featurizer(test_dataset, representation1), batch_size=batch_size)

    train_loader10 = GraphDataLoader(featurizer(train_dataset, representation10), batch_size=batch_size, shuffle=True)
    valid_loader10 = GraphDataLoader(featurizer(val_dataset, representation10), batch_size=batch_size)
    test_loader10 = GraphDataLoader(featurizer(test_dataset, representation10), batch_size=batch_size)

    train_loader = GraphDataLoader(featurizer(train_dataset, representationAll), batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(featurizer(val_dataset, representationAll), batch_size=batch_size)
    test_loader = GraphDataLoader(featurizer(test_dataset, representationAll), batch_size=batch_size)


    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader




def GetDataSDFHuman(path, y, smiles):
    dataset = PandasTools.LoadSDF(path)

    dataset = shuffle(dataset)
    
    dataset = dataset.drop(columns=['CdId', 'Field 2', 'Field 3', 'Field 5', 'Field 6', 'ID'])
    dataset = dataset.rename(columns={'Field 4': 'halflifetime_hr'})
    dataset = dataset.rename(columns={'ROMol': 'smiles'})
    dataset['halflifetime_hr'] = dataset['halflifetime_hr'].astype(float)
    dataset = dataset[dataset['halflifetime_hr'] < 15]

    dataset['smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(x))

    train_dataset = dataset[:int(len(dataset) * 0.7)]
    val_dataset = dataset[int(len(dataset) * 0.7): int(len(dataset) * 0.85)]
    test_dataset = dataset[int(len(dataset) * 0.85):]
    
    featurizer = ECFPFeaturizer(y_column=y, smiles_col=smiles)
    X_train, y_train = featurizer(train_dataset)
    X_valid, y_valid = featurizer(val_dataset)
    X_test, y_test = featurizer(test_dataset)

    batch_size = 64

    featurizer = GraphFeaturizer(y, smiles_col="smiles")

    train_loader1 = GraphDataLoader(featurizer(train_dataset, representation1), batch_size=batch_size, shuffle=True)
    valid_loader1 = GraphDataLoader(featurizer(val_dataset, representation1), batch_size=batch_size)
    test_loader1 = GraphDataLoader(featurizer(test_dataset, representation1), batch_size=batch_size)

    train_loader10 = GraphDataLoader(featurizer(train_dataset, representation10), batch_size=batch_size, shuffle=True)
    valid_loader10 = GraphDataLoader(featurizer(val_dataset, representation10), batch_size=batch_size)
    test_loader10 = GraphDataLoader(featurizer(test_dataset, representation10), batch_size=batch_size)

    train_loader = GraphDataLoader(featurizer(train_dataset, representationAll), batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(featurizer(val_dataset, representationAll), batch_size=batch_size)
    test_loader = GraphDataLoader(featurizer(test_dataset, representationAll), batch_size=batch_size)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader




def GetDataSDFRat(path, y, smiles):
    dataset = PandasTools.LoadSDF(path)

    dataset = shuffle(dataset)

    dataset = dataset.drop(columns=['CdId', 'Field 2', 'Field 3', 'ID', 'Field 5'])
    dataset = dataset.rename(columns={'Field 4': 'halflifetime_hr'})
    dataset = dataset.rename(columns={'ROMol': 'smiles'})
    dataset['halflifetime_hr'] = dataset['halflifetime_hr'].astype(float)
    dataset = dataset[dataset['halflifetime_hr'] < 15]
    
    dataset['smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(x))

    train_dataset = dataset[:int(len(dataset) * 0.7)]
    val_dataset = dataset[int(len(dataset) * 0.7): int(len(dataset) * 0.85)]
    test_dataset = dataset[int(len(dataset) * 0.85):]

    featurizer = ECFPFeaturizer(y_column=y, smiles_col=smiles)

    X_train, y_train = featurizer(train_dataset)
    X_valid, y_valid = featurizer(val_dataset)
    X_test, y_test = featurizer(test_dataset)

    batch_size = 64
    featurizer = GraphFeaturizer(y, smiles_col="smiles")
    
    train_loader1 = GraphDataLoader(featurizer(train_dataset, representation1), batch_size=batch_size, shuffle=True)
    valid_loader1 = GraphDataLoader(featurizer(val_dataset, representation1), batch_size=batch_size)
    test_loader1 = GraphDataLoader(featurizer(test_dataset, representation1), batch_size=batch_size)

    train_loader10 = GraphDataLoader(featurizer(train_dataset, representation10), batch_size=batch_size, shuffle=True)
    valid_loader10 = GraphDataLoader(featurizer(val_dataset, representation10), batch_size=batch_size)
    test_loader10 = GraphDataLoader(featurizer(test_dataset, representation10), batch_size=batch_size)

    train_loader = GraphDataLoader(featurizer(train_dataset, representationAll), batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(featurizer(val_dataset, representationAll), batch_size=batch_size)
    test_loader = GraphDataLoader(featurizer(test_dataset, representationAll), batch_size=batch_size)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader