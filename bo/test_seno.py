import sys
sys.path.append('../rxnft_vae')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import subprocess

import math, random, sys
from optparse import OptionParser
from collections import deque

from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score
#from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
#from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
#from vae import FTRXNVAE, set_batch_nodeID
#from mpn import MPN,PP,Discriminator
#from evaluate import Evaluator
import random
import numpy as np
import networkx as nx
from tdc import Oracle
from sparse_gp import SparseGP
import scipy.stats as sps
import sascorer
import pandas as pd



# Sample SMILES strings
smiles_list = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CCN(CC)C(=O)C1=CC=C(C=C1)N(C)C",
    "CC1=C(C(=O)NO)C(=O)C2=C(C1=O)N3CC4C(C3)C3(CC(C(C3)OC3CC(N(C)C)C(O)C(C)O3)O4)O2"
]

# Prepare SMILES as a single string, each enclosed in single quotes and separated by spaces
all_smiles = ' '.join([f"'{smile}'" for smile in smiles_list])

try:
    # Run the run_seno.sh script with all SMILES at once
    result = subprocess.run(f"./run_seno.sh {all_smiles}", capture_output=True, text=True, check=True, shell=True)
    
    # Split the output into individual scores
    scores = [float(score) for score in result.stdout.strip().split()]
    
    # Print results
    print("SMILES strings and their scores:")
    for smile, score in zip(smiles_list, scores):
        print(f"{smile}: {score}")

except subprocess.CalledProcessError as e:
    print(f"Error running run_seno.sh: {e}")
    print(f"stderr: {e.stderr}")
