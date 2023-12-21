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
from sklearn import StratifiedKFold

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
    



def GetDataCSV(path, y_column, smiles, seed):
    dataset = pd.read_csv(path)

    dataset = shuffle(dataset, random_state=seed)

    test_dataset = dataset[:len(dataset)*0.1]
    dataset = dataset[len(dataset)*0.1:]

    core_dataset_loaders = []
    subset_size = len(dataset) // 5

    core_dataset_loaders.append(dataset[:subset_size])
    core_dataset_loaders.append(dataset[subset_size:2*subset_size])
    core_dataset_loaders.append(dataset[2*subset_size:3*subset_size])
    core_dataset_loaders.append(dataset[3*subset_size:4*subset_size])
    core_dataset_loaders.append(dataset[4*subset_size:])
    
    featurizer = ECFPFeaturizer(y_column=y_column, smiles_col=smiles)
    X, y = featurizer(test_dataset)
    batch_size = 64
    featurizer = GraphFeaturizer(y_column, smiles_col=smiles)

    dataset_loaders1 = []
    dataset_loaders10 = []
    dataset_loaders = []

    for core_dataset in core_dataset_loaders:
        dataset_loaders1.append(GraphDataLoader(featurizer(core_dataset, representation1), batch_size=batch_size, shuffle=False))
        dataset_loaders10.append(GraphDataLoader(featurizer(core_dataset, representation10), batch_size=batch_size, shuffle=False))
        dataset_loaders.append(GraphDataLoader(featurizer(core_dataset, representationAll), batch_size=batch_size, shuffle=False))

    return X, y, test_dataset, dataset_loaders1, dataset_loaders10, dataset_loaders

    ### WYWALIĆ X, CHYBA NIE POTRZEBNY

def GetDataSDFHuman(path, y_column, smiles, seed):
    dataset = PandasTools.LoadSDF(path)

    dataset = shuffle(dataset, random_state=seed)
    
    dataset = dataset.drop(columns=['CdId', 'Field 2', 'Field 3', 'Field 5', 'Field 6', 'ID'])
    dataset = dataset.rename(columns={'Field 4': 'halflifetime_hr'})
    dataset = dataset.rename(columns={'ROMol': 'smiles'})
    dataset['halflifetime_hr'] = dataset['halflifetime_hr'].astype(float)
    dataset = dataset[dataset['halflifetime_hr'] < 15]

    dataset['smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(x))
    
    featurizer = ECFPFeaturizer(y_column=y_column, smiles_col=smiles)

    dataset = shuffle(dataset, random_state=seed)

    test_dataset = dataset[:len(dataset)*0.1]
    dataset = dataset[len(dataset)*0.1:]

    core_dataset_loaders = []
    subset_size = len(dataset) // 5

    core_dataset_loaders.append(dataset[:subset_size])
    core_dataset_loaders.append(dataset[subset_size:2*subset_size])
    core_dataset_loaders.append(dataset[2*subset_size:3*subset_size])
    core_dataset_loaders.append(dataset[3*subset_size:4*subset_size])
    core_dataset_loaders.append(dataset[4*subset_size:])

    X, y = featurizer(test_dataset)

    batch_size = 64

    dataset_loaders1 = []
    dataset_loaders10 = []
    dataset_loaders = []

    for core_dataset in core_dataset_loaders:
        dataset_loaders1.append(GraphDataLoader(featurizer(core_dataset, representation1), batch_size=batch_size, shuffle=False))
        dataset_loaders10.append(GraphDataLoader(featurizer(core_dataset, representation10), batch_size=batch_size, shuffle=False))
        dataset_loaders.append(GraphDataLoader(featurizer(core_dataset, representationAll), batch_size=batch_size, shuffle=False))

    return X, y, test_dataset, dataset_loaders1, dataset_loaders10, dataset_loaders




def GetDataSDFRat(path, y_column, smiles, seed):
    dataset = PandasTools.LoadSDF(path)

    dataset = shuffle(dataset, random_state=seed)

    dataset = dataset.drop(columns=['CdId', 'Field 2', 'Field 3', 'ID', 'Field 5'])
    dataset = dataset.rename(columns={'Field 4': 'halflifetime_hr'})
    dataset = dataset.rename(columns={'ROMol': 'smiles'})
    dataset['halflifetime_hr'] = dataset['halflifetime_hr'].astype(float)
    dataset = dataset[dataset['halflifetime_hr'] < 15]
    
    dataset['smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(x))


    test_dataset = dataset[:len(dataset)*0.1]
    dataset = dataset[len(dataset)*0.1:]

    core_dataset_loaders = []
    subset_size = len(dataset) // 5

    core_dataset_loaders.append(dataset[:subset_size])
    core_dataset_loaders.append(dataset[subset_size:2*subset_size])
    core_dataset_loaders.append(dataset[2*subset_size:3*subset_size])
    core_dataset_loaders.append(dataset[3*subset_size:4*subset_size])
    core_dataset_loaders.append(dataset[4*subset_size:])

    featurizer = ECFPFeaturizer(y_column=y_column, smiles_col=smiles)

    X, y = featurizer(test_dataset)

    batch_size = 64
    featurizer = GraphFeaturizer(y_column, smiles_col=smiles)
    
    dataset_loaders1 = []
    dataset_loaders10 = []
    dataset_loaders = []

    for core_dataset in core_dataset_loaders:
        dataset_loaders1.append(GraphDataLoader(featurizer(core_dataset, representation1), batch_size=batch_size, shuffle=False))
        dataset_loaders10.append(GraphDataLoader(featurizer(core_dataset, representation10), batch_size=batch_size, shuffle=False))
        dataset_loaders.append(GraphDataLoader(featurizer(core_dataset, representationAll), batch_size=batch_size, shuffle=False))

    return X, y, test_dataset, dataset_loaders1, dataset_loaders10, dataset_loaders
