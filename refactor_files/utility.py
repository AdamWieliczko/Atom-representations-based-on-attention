import torch


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def printProperties(atom):
    print("=========")
    print("GetDegree", atom.GetDegree())
    print("GetImplicitValence", atom.GetImplicitValence())
    print("GetAtomicNum", atom.GetAtomicNum())
    print("GetTotalNumHs", atom.GetTotalNumHs())
    print("GetNumImplicitHs", atom.GetNumImplicitHs())
    print("GetNeighbors", atom.GetNeighbors())
    print("GetNumExplicitHs", atom.GetNumExplicitHs())
    print("GetTotalDegree", atom.GetTotalDegree())
    print("GetTotalNumHs", atom.GetTotalNumHs())
    print("GetTotalValence", atom.GetTotalValence())


def attSequential(n_feats):
    return torch.nn.Sequential(torch.nn.Linear(n_feats, 1),
                       torch.nn.BatchNorm1d(1), torch.nn.ReLU(),
                       torch.nn.Linear(1, 1), torch.nn.ReLU())