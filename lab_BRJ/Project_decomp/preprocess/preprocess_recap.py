import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap, BRICS
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import re
from pathlib import Path


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

#===============================================================================================================
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
#===============================================================================================================


def remove_star_atoms(frag_mol):
    """Create a substructure by removing atoms with '*' symbols."""
    editable_mol = Chem.EditableMol(frag_mol)
    atoms_to_remove = [atom.GetIdx() for atom in frag_mol.GetAtoms() if atom.GetSymbol() == '*']
    
    # Remove atoms in reverse order to prevent index shifting issues
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(idx)
        
    return editable_mol.GetMol()

def recap_decompose(mol):
    """Perform RECAP decomposition and track the original atom indices for each substructure."""
    
    # Perform RECAP decomposition
    recap_tree = Recap.RecapDecompose(mol)
    recap_fragments = recap_tree.GetLeaves().keys()
    
    # Set atom map to original indices for each atom in the original molecule
    set_atom_map_numbers(mol)

    recap_indices = []
    labeled_mols = []  
    bonds_to_break_recap = set()  # Store broken bonds

    for frag_smiles in recap_fragments:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol:
            # Remove '*' atoms for clean substructure matching
            clean_frag_mol = remove_star_atoms(frag_mol)
            match = mol.GetSubstructMatch(clean_frag_mol)
            
            if match:
                # Assign original indices to matched atoms
                for atom, idx in zip(clean_frag_mol.GetAtoms(), match):
                    atom.SetAtomMapNum(mol.GetAtomWithIdx(idx).GetAtomMapNum())
                
                # Collect fragment indices and append to results
                frag_indices = [mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in match]
                recap_indices.append(sorted(frag_indices))
                labeled_mols.append(clean_frag_mol)
                
                # Identify bonds broken during decomposition
                frag_indices_set = set(frag_indices)
                for idx in frag_indices_set:
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if neighbor_idx not in frag_indices_set:
                            bonds_to_break_recap.add((min(idx, neighbor_idx), max(idx, neighbor_idx)))
            else:
                # If no match, append empty index list
                recap_indices.append([])
                labeled_mols.append(clean_frag_mol)

    return recap_indices, labeled_mols, sorted(list(bonds_to_break_recap))

def process_smiles_with_recap(smiles_list):
    """
    SMILES 데이터를 처리하고, RECAP 분해에서 에러가 발생한 인덱스를 저장하며,
    성공적으로 분해된 모든 조각을 집합으로 반환하는 함수.

    Args:
        smiles_list (list): SMILES 문자열의 리스트.
        recap_decompose (function): RECAP 분해를 수행하는 함수.
        convert_smiles_to_mol_objects (function): SMILES 문자열을 Molecule 객체로 변환하는 함수.

    Returns:
        tuple: (recap_all_frag, recap_error_indices)
            - recap_all_frag (set): 성공적으로 처리된 Molecule 조각들의 집합.
            - recap_error_indices (list): 에러가 발생한 인덱스 리스트.
    """
    recap_error_indices = []
    recap_all_frag = set()

    for i in range(len(smiles_list)):
        try:
            recap_indices, recap_mols, bonds_to_break_recap = recap_decompose(Chem.MolFromSmiles(smiles_list[i]))
            recap_mols = [Chem.MolToSmiles(mol) for mol in recap_mols]
            convert_recap_mols = set(convert_smiles_to_mol_objects(recap_mols))
            recap_all_frag = recap_all_frag.union(convert_recap_mols)
        except Exception as e:
            recap_error_indices.append(i)  # 에러 발생 인덱스 저장

    return recap_all_frag, recap_error_indices


# Main Function
def process_smiles(input_file):
    data = pd.read_csv(input_file)
    smiles_list = data['smiles']
    recap_all_frag, _ = process_smiles_with_recap(smiles_list)

    # One-hot encoding
    recap_encoded = one_hot_encode_all(recap_all_frag)

    # Convert the one-hot encoding dictionary to a pandas DataFrame
    recap_encoded_df = pd.DataFrame.from_dict(recap_encoded, orient="index")
    recap_encoded_df.reset_index(inplace=True)
    recap_encoded_df.columns = ["Fragment", *[f"Feature_{i}" for i in range(recap_encoded_df.shape[1] - 1)]]

    # Save to CSV
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    recap_encoded_df.to_csv(output_dir / "recap_encoded.csv", index=False)

    print(f"Results saved to {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SMILES for RECAP decomposition.")
    parser.add_argument("input_file", help="Path to the input CSV file containing SMILES strings.")
    args = parser.parse_args()
    process_smiles(args.input_file)
