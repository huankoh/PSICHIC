import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, virtual_screening
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *
# Preprocessing
from utils import protein_init, ligand_init
# Model
from models.net import net
# Config
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

## Device and batch size
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--trained_model_path',type=str, default='trained_weights/PDBv2020_PSICHIC')
parser.add_argument('--batch_size', type=int, default=16)
## Data and Pre-processing
parser.add_argument('--screenfile', type=str, default='dataset/pdb2020/test.csv', help='csv file')
parser.add_argument('--result_path', type=str,default='FinalScreen_pdb2020',help='path to save results')
parser.add_argument('--save_interpret', type=bool,default=True,help='Save interpretation from PSICHIC?')

args = parser.parse_args()


with open(os.path.join(args.trained_model_path,'config.json'),'r') as f:
    config = json.load(f)

print("Screening the csv file: {}".format(args.screenfile))
# device
device = torch.device(args.device)


if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

print(args)
with open(os.path.join(args.result_path, 'screening_params.txt'), 'w') as f:
    f.write(str(args))

degree_dict = torch.load(os.path.join(args.trained_model_path,'degree.pt'))
param_dict = os.path.join(args.trained_model_path,'model.pt')
mol_deg, prot_deg = degree_dict['ligand_deg'],degree_dict['protein_deg']

model = net(mol_deg, prot_deg,
            # MOLECULE
            mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'], 
            prot_evo_channels=config['params']['prot_evo_channels'],
            hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'], 
            post_layers=config['params']['post_layers'],aggregators=config['params']['aggregators'], 
            scalers=config['params']['scalers'],total_layer=config['params']['total_layer'],                
            K = config['params']['K'],heads=config['params']['heads'], 
            dropout=config['params']['dropout'],
            dropout_attn_score=config['params']['dropout_attn_score'],
            # output
            regression_head=config['tasks']['regression_task'],
            classification_head=config['tasks']['classification_task'] ,
            multiclassification_head=config['tasks']['mclassification_task'],
            device=device).to(device)
model.reset_parameters()    
model.load_state_dict(torch.load(param_dict,map_location=args.device))


screen_df = pd.read_csv(os.path.join(args.screenfile))
protein_seqs = screen_df['Protein'].unique().tolist()
print('Initialising protein sequence to Protein Graph')
protein_dict = protein_init(protein_seqs)
ligand_smiles = screen_df['Ligand'].unique().tolist()
print('Initialising ligand SMILES to Ligand Graph')
ligand_dict = ligand_init(ligand_smiles)
torch.cuda.empty_cache()
## drop any invalid smiles
screen_df = screen_df[screen_df['Ligand'].isin(list(ligand_dict.keys()))].reset_index(drop=True)
screen_dataset = ProteinMoleculeDataset(screen_df, ligand_dict, protein_dict, device=args.device)
screen_loader = DataLoader(screen_dataset, batch_size=args.batch_size, shuffle=False,
                            follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

print("Screening starts now!")
screen_df = virtual_screening(screen_df, model, screen_loader,
                 result_path=os.path.join(args.result_path, "interpretation_result"), save_interpret=args.save_interpret, 
                 ligand_dict=ligand_dict, device=args.device)

screen_df.to_csv(os.path.join(args.result_path,'screening.csv'),index=False)
print('Screening completed and saved to {}'.format(args.result_path))