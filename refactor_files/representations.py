from rdkit import Chem
from utility import one_of_k_encoding, one_of_k_encoding_unk

def defaultRepresentation(atom):
    return one_of_k_encoding_unk(atom.GetAtomicNum(), range(11)) + one_of_k_encoding(
                    atom.GetDegree(), range(11)
                ) + one_of_k_encoding_unk(
                    atom.GetImplicitValence(), range(11)
                ) + [atom.GetIsAromatic()] + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(), range(11)
                ) + [atom.GetNumImplicitHs(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.IsInRing()] # TODO: Add atom features as a list, you can use one_of_k_encodings defined above


def representation1(atom):
    return one_of_k_encoding_unk(atom.GetAtomicNum(), range(12)) + one_of_k_encoding_unk(
                    atom.GetDegree(), range(6)) + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(), range(5)
                ) + [atom.GetFormalCharge(), atom.IsInRing(), atom.GetIsAromatic()]

def representation10(atom):
    return one_of_k_encoding_unk(atom.GetAtomicNum(), range(12)) + one_of_k_encoding_unk(
                    atom.GetDegree(), range(6)) + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(), range(5)
                ) + [atom.IsInRing(), atom.GetIsAromatic()]

def representationAll(atom):
    return one_of_k_encoding_unk(atom.GetAtomicNum(), range(12)) + one_of_k_encoding_unk(
                    atom.GetDegree(), range(6)) + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(), range(5)) + one_of_k_encoding_unk(
                    atom.GetImplicitValence(), range(6))  + one_of_k_encoding_unk(
                    atom.GetHybridization(),
                    [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2
                    ]
                ) + [atom.GetFormalCharge(), atom.IsInRing(), atom.GetIsAromatic()
                     ] + [atom.GetNumRadicalElectrons()]