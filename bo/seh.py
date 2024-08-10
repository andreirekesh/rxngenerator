from bengio2021flow import load_original_model, mol2graph
from typing import List, Optional
import torch
import torch_geometric.data as gd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from pydantic import Field
from pydantic.dataclasses import dataclass
import os
import sys


@dataclass
class Parameters:
    model = load_original_model()
    device: str = "cuda"
    log_dir: str = "./logs_seh"
    beta: int = 8


class SEHProxy():
    def __init__(self, params: Parameters):
        self.device = params.device
        self.model = params.model
        self.log_dir = params.log_dir
        self.beta = params.beta
        self.model.to(self.device)

    def __call__(self, smiles: List[str]) -> np.array:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        graphs = [mol2graph(i) for i in mols]
        smiles_to_save = smiles
        batch = gd.Batch.from_data_list(graphs)
        batch.to(self.device)
        self.model.to(self.device)
        raw_scores = self.model(batch).reshape((-1,)).data.cpu()
        raw_scores[raw_scores.isnan()] = 0
        scores_to_save = list(raw_scores)
        #with open(os.path.join(self.log_dir, "visited.txt"), 'a') as file:
        #    # Write each molecule and its score to the file
        #    for molecule, score in zip(smiles_to_save, scores_to_save):
        #        file.write(f"{molecule}, {score}\n")

        transformed_scores = raw_scores.clip(1e-4, 100).reshape((-1,))
        #print(f"Proxy Mean: {raw_scores.mean()}, Proxy Max: {raw_scores.max()}, Mean Reward: {transformed_scores.mean()}, Max Reward: {transformed_scores.max()}")
        return list(np.array(transformed_scores, dtype=float))

def main():
    # Get the list of SMILES strings from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python my_script.py <smiles1> <smiles2> ...")
        sys.exit(1)

    smiles_list = sys.argv[1:]
    params = Parameters()
    proxy = SEHProxy(params)
    #print(smiles_list)
    results = proxy(smiles_list)
    
    # Print or save the results as needed
    print(" ".join(map(str, results)))
    #print('hi')

def main_old():
    import pandas as pd
    import numpy as np

    # Read the file and extract SMILES
    smiles_list = []
    with open('../data/data.txt', 'r') as file:
        for line in file:
            smiles = line.split('*')[0].strip()
            smiles_list.append(smiles)

    # Initialize the SEHProxy
    params = Parameters()
    proxy = SEHProxy(params)

    # Process SMILES in batches
    batch_size = 1000
    seh_scores = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_scores = proxy(batch)
        seh_scores.extend(batch_scores)
        print(f"Processed batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}")

    # Read existing CSV file or create new DataFrame
    try:
        df = pd.read_csv('seno_results.csv')
    except FileNotFoundError:
        df = pd.DataFrame({'SMILES': smiles_list})

    # Add SEH scores as a new column
    df['SEH_Score'] = seh_scores

    # Save updated DataFrame to CSV
    df.to_csv('seno_results.csv', index=False)

    print(f"Processed {len(smiles_list)} SMILES. SEH scores added to seno_results.csv")

if __name__ == "__main__":
    main()