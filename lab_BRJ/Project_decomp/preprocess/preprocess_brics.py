import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap, BRICS
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import re
from pathlib import Path
from rdkit.Chem.rdchem import RWMol


# Helper Functions
def set_atom_map_numbers(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def remove_atom_mapping(smiles):
    return re.sub(r'\:\d+\]', ']', smiles)


def convert_smiles_to_mol_objects(smiles_list):
    mol_objects = []
    for smiles in smiles_list:
        cleaned_smiles = remove_atom_mapping(smiles)
        try:
            mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=False)
            if mol is not None:
                smiles_with_aromaticity = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
                mol_objects.append(smiles_with_aromaticity)
        except Exception as e:
            print(f"Failed to create Mol object from: {cleaned_smiles}, error: {e}")
    return mol_objects

#==============================================================================================================
def one_hot_encode_all(frag):
    frag_list = list(frag)
    value_to_index = {value: idx for idx, value in enumerate(frag_list)}
    
    # 원핫 인코딩 생성 함수
    def one_hot_encode(value, value_to_index, length):
        encoding = [0] * length
        index = value_to_index.get(value)
        if index is not None:
            encoding[index] = 1
        return encoding

    # 각 값에 대해 원핫 인코딩 수행
    one_hot_encodings = {value: one_hot_encode(value, value_to_index, len(frag_list)) for value in frag_list}
    return one_hot_encodings
#==============================================================================================================


def brics_decompose(mol):
    # BRICS decomposition
    rw_mol_brics = RWMol(mol)
    bonds_to_break_brics = list(BRICS.FindBRICSBonds(mol))

    # Set atom map to original indices for each atom in the original molecule
    set_atom_map_numbers(mol)

    for bond in bonds_to_break_brics:
        atom1, atom2 = bond[0]
        rw_mol_brics.RemoveBond(atom1, atom2)

    brics_fragments = Chem.GetMolFrags(rw_mol_brics, asMols=True, sanitizeFrags=False)
    brics_fragment_indices = []
    brics_mols = []
    for frag in brics_fragments:
        indices = [atom.GetAtomMapNum() for atom in frag.GetAtoms()]
        brics_fragment_indices.append(indices)
        brics_mols.append(frag)  # Directly append the fragment Mol object

    return brics_fragment_indices, brics_mols, bonds_to_break_brics

def process_smiles_with_brics(smiles_list):
    # 에러 발생 인덱스를 저장할 리스트
    brics_error_indices = []
    brics_all_frag = set()

    for i in range(len(smiles_list)):
        try:
            brics_indices, brics_mols, bonds_to_break_brics = brics_decompose(Chem.MolFromSmiles(smiles_list[i]))
            brics_mols = [Chem.MolToSmiles(mol) for mol in brics_mols]
            convert_brics_mols = set(convert_smiles_to_mol_objects(brics_mols))
            brics_all_frag = brics_all_frag.union(convert_brics_mols)

        except Exception as e:
            brics_error_indices.append(i)  # 에러 발생 인덱스 저장
    
    return brics_all_frag, brics_error_indices

# Main Function
def process_smiles(input_file):
    data = pd.read_csv(input_file)
    smiles_list = data['smiles']
    brics_all_frag, _ = process_smiles_with_brics(smiles_list)

    # One-hot encoding
    brics_encoded = one_hot_encode_all(brics_all_frag)

    # Convert the one-hot encoding dictionary to a pandas DataFrame
    brics_encoded_df = pd.DataFrame.from_dict(brics_encoded, orient="index")
    brics_encoded_df.reset_index(inplace=True)
    brics_encoded_df.columns = ["Fragment", *[f"Feature_{i}" for i in range(brics_encoded_df.shape[1] - 1)]]

    # Save to CSV
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    brics_encoded_df.to_csv(output_dir / "brics_encoded.csv", index=False)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SMILES for BRICS.")
    parser.add_argument("input_file", help="Path to the input CSV file containing SMILES strings.")
    args = parser.parse_args()
    process_smiles(args.input_file)
