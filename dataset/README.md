# PSICHIC Dataset

Below describe all the datasets that are available on Google Drive ([Dataset](https://drive.google.com/drive/folders/1ZRpnwXtllCP89hjhfDuPivBlarBIXnmu?usp=sharing)).

## General Format 
In general, the CSV files for training, validation, and test splits will have a specific format. These datasets originally come in a single CSV file containing all annotated sequence data (protein-ligand pairs). For benchmark evaluations, we consistently adhere to established split settings. This is done either by using dataset IDs (e.g., PDB IDs in PDBBind datasets) or by employing the same splitting strategy as in previous works. We also provide a Jupyter notebook in this folder to demonstrate how we divide the datasets based on random split, unseen ligand scaffold split, and/or unseen protein split.
__Binding Affinity Regression__

| Protein | Ligand | regression_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 6.7 | 
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 4.0 |
|...|...| ...|
|TACGTACG | CCO | 8.1 | 

__Binary Interaction Classification__

| Protein | Ligand | classification_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 1 | 
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 0 |
|...|...| ...|
|TACGTACG | CCO | 1 | 

__Functional Effect Classification (Three-way Classification)__

| Protein | Ligand | multiclass_label  | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | -1 | 
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 0 |
|...|...| ...|
|TACGTACG | CCO | 1 | 

_note, antagonist is represented as -1, non-binder as 0 and agonist as 1._

__Multi Task PSICHIC__

| Protein | Ligand | regression_label | multiclass_label | 
|:----------:|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 6.7 | -1 |  # antagonist
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 4.0 | 0 | # non-binder
|...|...| ...|
|TACGTACG | CCO | 8.1 | 1 | # agonist

## Dataset Description

|  **Dataset Path**  | **Task** | **Type**                                       |                          **Description**                           |
| :--------: |:----: | :--------------------------------------------: | :----------------------------------------------------------: |
|  **PDBBindv2016**  | Binding Affinity Regression | Benchmark Evaluation (Effectiveness) | Each sample in the PDBBind v2016 dataset is a complex, but we extracted the sequence data with substantial information loss to yield a protein-ligand sequence pair. We maintained the same split setting used in a previous study, where the refined set (excluding the core set) is treated as  training (train.csv) and validation (valid.csv) sets, while the core set (complexes with the highest resolution) is treated as the test set (test.csv). Other than 'Protein', 'Ligand', and 'regression_label', the CSV files have a column 'ID' that represents the PDB ID ('id_' + PDB ID), and a column 'Target_Chain' to represent the chain to which the amino acid position belongs.
|  **PDBBindv2020**  |  Binding Affinity Regression| Benchmark Evaluation (Effectiveness) | Similarly, each Sample of PDBBind v2020 dataset is a complex, but we extracted the sequence data with substantial information loss to gives us protein-ligand sequence pair. We maintained the temporal split setting used in previous study where data points before 2019 are training (train.csv) and validation (valid.csv) sets, while during and after 2019 are test sets (test.csv). Other than 'Protein', 'Ligand', and 'regression_label', the CSV files have a column 'ID' that represents the PDB ID ('id_' + PDB ID), and a column 'Target_Chain' to represent the chain to which the amino acid position belongs.
|  **PDBBind2020TestSet_StructuralResolution**  |  Binding Affinity Regression | Robustness Evaluation (Effectiveness) | We extracted the resolution of complexes in PDBBind2020 TestSet to evaluate whether a method is sensitive to the resolution (e.g., higher prediction errors, when resolution deteriorates). The file is a json file with keys being the PDB IDs and values being the numerical value of PDB resolution.
| **PDBBind2020TestSet_InSilicoStructures** | Binding Affinity Regreesion | Robustness Evaluation (Effectiveness) | Used for evaluating structure-based and complex-based methods where we generated 3D protein structures using AlphaFold2. The folders contain 3D structures generated using AlphaFold2 and ESMFold, as well as complex structures generated using DiffDock on the AlphaFold2 and ESMFold structures
| **Human/random** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand Pairs in Human Sequence-based Dataset are randomly assigned as training (train.csv), validation (valid.csv) and test sets (test.csv) with a ratio of 7:1:2.
| **Human/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the Human sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen ligand scaffold split method. This approach first computes all unique scaffolds in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 ratio as possible.
| **Human/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the Human sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen protein method. This approach first find the unique proteins in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 datapoint ratio as possible.
| **BioSNAP/random** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand Pairs in BioSNAP Sequence-based Dataset are randomly assigned as training (train.csv), validation (valid.csv) and test sets (test.csv) with a ratio of 7:1:2.
| **BioSNAP/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the BioSNAP sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen ligand scaffold split method. This approach first computes all unique scaffolds in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 ratio as possible.
| **BioSNAP/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the BioSNAP sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen protein method. This approach first find the unique proteins in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 datapoint ratio as possible.
| **BindingDB/random** | Binary Interaction Classification |  Benchmark Evaluation (Generalizability) | Protein-Ligand Pairs in BindingDB Sequence-based Dataset are randomly assigned as training (train.csv), validation (valid.csv) and test sets (test.csv) with a ratio of 7:1:2.
| **BindingDB/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the BindingDB sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen ligand scaffold split method. This approach first computes all unique scaffolds in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 ratio as possible.
| **BindingDB/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand pairs in the BindingDB sequence-based dataset are carefully split with a 7:1:2 ratio using the unseen protein method. This approach first find the unique proteins in the dataset, and then assigns them to training (train.csv), validation (valid.csv), and test sets (test.csv), aiming to achieve as close to a 7:1:2 datapoint ratio as possible.
| **ProteinLigandFunctionalEffect** | Functional Effect Classification | Benchmark Evaluation (Generalizability) | Protein-Ligand Pairs in Functional Effect Sequence-based Dataset are randomly assigned as training (train.csv), validation (valid.csv) and test sets (test.csv) with a ratio of 7:1:2.
| **LargeScaleInteractionDataset** | Binding Affinity Regression + Functional Effect Classification | Real-world Deployment (PSICHIC<sub>XL</sub>) | The dataset (train.csv) was used to train PSICHIC<sub>XL</sub>. The test file (test.csv) is only used for validation to track losses and monitor the model's training progress. The CSV columns include 'Protein' for protein sequences, 'Ligand' for ligand SMILES, 'regression_label' for binding affinity labels, and 'multiclass_label' for functional effect classes (where -1 indicates an antagonist, 0 indicates a non-binder, 1 indicates an agonist, 999 indicates a binder but it's unclear if it's an agonist/antagonist/other, and NaN indicates an unlabeled datapoint). 'key' indicates a unique key for the pair, 'activity_label' indicates a 1/0 interaction binary class (Binder/Non-Binder), and 'target_activity' indicates the protein ID plus whether the ligand binds to the target. 'scaffold' indicates the ligand's scaffold, 'target_activity_number_unique_scaffold' indicates the number of unique scaffold ligands' datapoints for the given target_activity, and 'target_activity_weight' indicates the square root of the number of scaffolds per target_activity. 'joint_space' indicates the combination of target_activity and ligand scaffold, 'joint_space_count' indicates the number of ligands with the same scaffold per target_activity, 'pretrain_sampling_weight' for relative final weight assigned to the datapoint for training, 'Uniprot_ID' indicates the protein target's UniProt ID, and 'Functional_Effect_Class' indicates the functional effect classes in a descriptive manner.
| **A1R_FineTune** | Binding Affinity Regression + Functional Effect Classification | Real-world Deployment (PSICHIC<sub>A<sub>1</sub>R</sub>) | A subset of dA<sub>1</sub>R-related datapoints within the LargeScaleInteractionDataset are extracted to fine-tune PSICHIC<sub>XL</sub> into PSICHIC<sub>A<sub>1</sub>R</sub> (finetuning.ipynb notebook is provided in the A1R_FineTune folder for reproducing the extraction).
