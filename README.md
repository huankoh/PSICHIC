## PSICHIC: physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data [[Preprint](https://www.biorxiv.org/content/10.1101/2023.09.17.558145v1)]

<img src="image/PSICHIC.jpg" width="500"/>

## Environment Setup
Currently, PSICHIC is validated for use on MacOS (OSX) and Linux. We recommend installation via conda, or even better, using the faster mamba package and environment manager. Mamba can be installed with the command ``conda install mamba -n base -c conda-forge``. For setup using either conda or mamba, please refer to the relevant code line provided below.

```
## OSX 
conda env create -f environment_osx.yml  # if mamba, use ``mamba env create -f environment_osx.yml``
## LINUX CPU
conda env create -f environment_cpu.yml  # if mamba, use ``mamba env create -f environment_cpu.yml``
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

## LINUX GPU
conda env create -f environment_gpu.yml # if mamba, use ``mamba env create -f environment_gpu.yml``
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

```
_The yml files for Linux should also work for Windows but this is not validated yet due to travel commitments, we will do so ASAP (by 15 Feb 24) and will update this section accordingly._ 

Alternatively, command lines that can be helpful in setting up the environment (tested on linux with python 3.8). 
```
conda create --name psichic_fp python=3.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge rdkit==2022.09.5
pip install scipy biopython pandas biopandas timeout_decorator py3Dmol umap-learn plotly mplcursors lifelines reprint
pip install "fair-esm"
```

## Virtual Screening Platform <a href="https://colab.research.google.com/github/huankoh/PSICHIC/blob/main/PSICHIC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- **Only Sequence Data**: Protein Sequence + Ligand SMILES pairs is all you need.
- **Quick Screening**: Up to 100K compounds in an hour.
- **Deep Analysis**: Uncover molecular insights with PSICHIC-powered pharmacophore and targeted mutagenesis analysis.

_Start exploring. Your next discovery could be just clicks away!_

**UPDATE:** We now included selectivity online platform (beta version) in selectivity subfolder that demonstrate how PSICHIC can be used for selectivity profiling.

## BYO-PSICHIC with Annotated Sequence Data 

Create a train, valid and test csv file in a datafolder (for examples, see the dataset folder). The datafolder should contain at least a train.csv and test.csv file. Depending on your annotated labels, you want to use ``--regression_task True`` if it is a continuous value label (e.g., binding affinity), ``--classification_task True`` if it is a binary class label (e.g., presence of interaction) and ``--mclassification_task C`` where C represents the number of classes in your multiclass labels (e.g., 3 if you are using our protein-ligand functional response dataset). Note, you can have a dataset with multiple label types and we will train PSICHIC on predicting multiple protein-ligand interaction properties (see PSICHIC-MultiTask below)
```
python main.py --datafolder annotated_folder --result_path result/annotated_result --regression_task True 
```

BYO-PSICHIC using a benchmark dataset, for example, the PDBBind v2020 benchmark:
```
python main.py --datafolder pdb2020 --result_path result/PDB2020_BENCHMARK --regression_task True 
```
Model and optimizer configurations are consistent across all benchmark datasets, except PDBBind v2016 where you want to change the optimizer's number of training iterations, betas and eps to 30000, "(0.9,0.99)" and 1e-5 respectively, i.e. add to the commandline: ``--total_iters 30000 --betas "(0.9,0.99)" --eps 1e-5``. For binary classification task, replace ``--regression_task True`` to ``--classification_task True``. For protein functional effect dataset, replace ``--regression_task True`` to ``--mclassification_task 3``. Feel free to adjust the model hyperparameters in the config.json file, let us know if you find any interesting results!


## Dataset Structure and BYO Formatting Guidelines
All datasets referenced in our manuscript are available on Google Drive ([Dataset](https://drive.google.com/drive/folders/1ZRpnwXtllCP89hjhfDuPivBlarBIXnmu?usp=sharing)). For the datasets used in the benchmark evaluation of PSICHIC, we have train, valid, and test CSV files that have been created based on established split settings. A separate README.md in the dataset section is dedicated to explaining the purpose of each dataset in the Google Drive Link (this is similar to Extended Data Table 1 in our manuscript). 

BYO-PSICHIC Dataset: Each file should look something like this if you are interested in training BYO-PSICHIC. A validation CSV file is not required if you don't have one, for instance, if you plan to apply the results in external experiments.

__Binding Affinity Regression__

| Protein | Ligand | regression_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 6.7 | 
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 4.0 |
|...|...| ...|
|TACGTACG | CCO | 8.1 | 

```
python main.py --datafolder BYO_DATASET --result_path BYO_RESULT --regression_task True 
```

__Binary Interaction Classification__

| Protein | Ligand | classification_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 1 | 
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 0 |
|...|...| ...|
|TACGTACG | CCO | 1 | 

```
python main.py --datafolder BYO_DATASET --result_path BYO_RESULT --classification_task True
```

__Functional Effect Classification (Three-way Classification)__

| Protein | Ligand | multiclass_label | 
|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | -1 |  # antagonist
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 0 | # non-binder
|...|...| ...|
|TACGTACG | CCO | 1 | # agonist

```
python main.py --datafolder BYO_DATASET --result_path BYO_RESULT --mclassification_task 3
```

__Multi Task PSICHIC__

| Protein | Ligand | regression_label | multiclass_label | 
|:----------:|:----------:|:----------:|:----------:|
| ATCGATCG....  | C1CCCCC1  | 6.7 | -1 |  # antagonist
| GCTAGCTA....  | O=C(C)Oc1ccccc1C(=O)O | 4.0 | 0 | # non-binder
|...|...| ...|
|TACGTACG | CCO | 8.1 | 1 | # agonist

```
python main.py --datafolder BYO_DATASET --result_path BYO_RESULT --regression_task True --mclassification_task 3
```

**Strategically Split Your Dataset?** Jupyter notebook in dataset folder is available to illustrate how we perform random splits, unseen protein splits, and unseen ligand scaffold splits to evaluate the generalizability of PSICHIC or other methods. This can be useful in evaluating whether the BYO-PSICHIC works on your annotated sequence data.
 
## PSICHIC<sub>XL</sub>: Multitask Prediction Training on Large-scale Interaction Dataset
The PSICHIC<sub>XL</sub> was previously referred to as the pre-trained multi-task PSICHIC. The PSICHIC<sub>A1R</sub> was previously referred to as the fine-tuned multi-task PSICHIC. We changed the name to clarify that PSICHIC<sub>XL</sub> can be used as is without any additional training. However, PSICHIC<sub>XL</sub> can potentially improve its ranking capabilities in virtual screening when fine-tuned on data specific to a protein target, e.g., the PSICHIC<sub>A<sub>1</sub>R</sub> we show below using A<sub>1</sub>R-related data.

### Training PSICHIC<sub>XL</sub> (AKA Pre-trained PSICHIC in Preprint)
```
python main.py --datafolder dataset/large_scale_interaction_dataset --result_path PSICHIC_MultiTask_Pretrain --lrate 1e-5 --sampling_col pretrain_sampling_weight --regression_task True --mclassification_task 3 --total_iters 300000 --evaluate_step 25000
```
### Fine-tune PSICHIC<sub>XL</sub> into PSICHIC<sub>A<sub>1</sub>R</sub> (AKA Fine-tuned PSICHIC in Preprint)
We finetune only the application layers of PSICHIC<sub>XL</sub> for 1000 iteration on A<sub>1</sub>R-related protein using the following command:
```
python main.py --regression_task True --mclassification_task 3 --datafolder dataset/A1R_FineTune --result_path PSICHIC_A1R_FineTune --lrate 1e-5 --total_iters 1000 --finetune_modules "['reg_out','mcls_out']" --trained_model_path trained_weights/multitask_PSICHIC
```
We have renamed the PSICHIC version trained on the extensive interaction dataset as PSICHIC<sub>XL</sub>, and the subset focusing on A<sub>1</sub>R data as PSICHIC<sub>A<sub>1</sub>R</sub>. Previously, PSICHIC<sub>XL</sub> and PSICHIC<sub>A<sub>1</sub>R</sub> were known as pre-trained PSICHIC and fine-tuned PSICHIC, respectively. This change more accurately reflects PSICHIC<sub>XL</sub>'s broad applicability and PSICHIC<sub>A<sub>1</sub>R</sub>'s specific emphasis on A1R.

For any other proteins, you can filter out irrelevant proteins and the non-binders in large-scale interaction dataset to apply PSICHIC for other experiments.

## References

For more information, please refer to our work: 

```
PSICHIC: physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data
Huan Yee Koh, Anh T.N. Nguyen, Shirui Pan, Lauren T. May, Geoffrey I. Webb
bioRxiv 2023.09.17.558145; doi: https://doi.org/10.1101/2023.09.17.558145
```
