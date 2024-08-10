from typing import List
from rdkit import Chem
import torch
import numpy as np
from gneprop.rewards import load_model, predict
from wurlitzer import pipes
from pydantic import Field
from pydantic.dataclasses import dataclass
import os
import sys

@dataclass
class Parameters:
    batch_size: int = 256
    checkpoint_path: str = "/workspace/Tyers/benchmark/Mpro-GFN/src/gflownet/gneprop_weights/epoch=31-step=2272.ckpt"
    log_dir = "./logs_seno"


class SenoProxy():
    def __init__(self, params: Parameters):
        self.batch_size = params.batch_size
        self.model = load_model(params.checkpoint_path)
        self.log_dir = params.log_dir

    def __call__(self, smiles: List[str]) -> np.array:
        try:
            with pipes():
                scores = (
                    predict(
                        self.model,
                        smiles,
                        batch_size=self.batch_size,
                        gpus=1,
                    )
                    * 100
                ).tolist()
        except Exception as e:
            print(f"GNEProp Score Exception: {e}")
            scores = [0] * len(smiles)
            
        #with open(os.path.join(self.log_dir, "visited.txt"), 'a') as file:
        #    # Write each molecule and its score to the file
        #    for molecule, score in zip(smiles, scores):
        #        file.write(f"{molecule}, {score}\n")

        return list(np.array(scores, dtype=float))

def main():
    # Get the list of SMILES strings from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python my_script.py <smiles1> <smiles2> ...")
        sys.exit(1)

    smiles_list = sys.argv[1:]
    params = Parameters()
    proxy = SenoProxy(params)
    #print(smiles_list)
    results = proxy(smiles_list)
    
    # Print or save the results as needed
    print(" ".join(map(str, results)))
    #print('hi')

def main_old():
    # Read the file and extract SMILES
    smiles_list = []
    with open('../data/data.txt', 'r') as file:
        for line in file:
            smiles = line.split('*')[0].strip()
            smiles_list.append(smiles)

    # Initialize and run the SenoProxy
    params = Parameters()
    proxy = SenoProxy(params)
    scores = proxy(smiles_list)

    # Save SMILES-score pairs to CSV
    import csv
    
    with open('seno_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILES', 'Score'])  # Write header
        for smiles, score in zip(smiles_list, scores):
            writer.writerow([smiles, score])

    print(f"Processed {len(smiles_list)} SMILES. Results saved to seno_results.csv")


if __name__ == "__main__":
    main()