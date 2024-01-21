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

| Protein | Ligand | mclassification_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | -1 |  # antagonist
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 0 | # non-binder
|...|...| ...|
|TACGTACG | CCO | 1 | # agonist

__Multi Task PSICHIC__

| Protein | Ligand | regression_label | mclassification_label | 
|:----------:|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 6.7 | -1 |  # antagonist
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 4.0 | 0 | # non-binder
|...|...| ...|
|TACGTACG | CCO | 8.1 | 1 | # agonist

## Dataset Description

|  **Dataset Path**  | **Task** | **Type**                                       |                          **Description**                           |
| :--------: |:----: | :--------------------------------------------: | :----------------------------------------------------------: |
|  **PDBBindv2016**  | Binding Affinity Regression | Benchmark Evaluation (Effectiveness) | PDBBind v2016 gives us protein-ligand
|  **PDBBindv2020**  |  Binding Affinity Regression| Benchmark Evaluation (Effectiveness) | ..
|  **PDBBind2020TestSet_StructuralResolution**  |  Binding Affinity Regression| Robustness Evaluation (Effectiveness) | ..
| **PDBBind2020TestSet_InSilicoStructures** | Binding Affinity Regreesion | Robustness Evaluation (Effectiveness) | Used for evaluating structure-based and complex-based methods
| **Human/random** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **Human/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **Human/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **BioSNAP/random** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **BioSNAP/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **BioSNAP/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **BindingDB/random** | Binary Interaction Classification |  Benchmark Evaluation (Generalizability) | ...
| **BindingDB/scaffold** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **BindingDB/protein** | Binary Interaction Classification | Benchmark Evaluation (Generalizability) | ...
| **ProteinLigandFunctionalEffect** | Functional Effect Classification | Benchmark Evaluation (Generalizability) | ...
| **LargeScaleInteractionDataset** | Binding Affinity Regression + Functional Effect Classification | Real-world Deployment (PSICHIC<sub>XL</sub>) | ...
| **A1R_FineTune** | Binding Affinity Regression + Functional Effect Classification | Real-world Deployment (PSICHIC<sub>A<sub>1</sub>R</sub>) | ...
