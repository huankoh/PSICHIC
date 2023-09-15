import torch
from utils.metrics import evaluate_cls, evaluate_mcls, evaluate_reg
import json 
from utils import unbatch
from reprint import output
import math
import os
import sys

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, lrate, min_lrate, wdecay, betas, eps, amsgrad, clip, steps_per_epoch, num_epochs, total_iters, 
                warmup_iters=2000, lr_decay_iters=None, schedule_lr=True, regression_weight=1, 
                classification_weight=1, multiclassification_weight=1, evaluate_metric='rmse', 
                result_path='', runid=0, device='cuda:0', skip_test_during_train=False,
                finetune_modules=None):
                
        self.model = model
        self.model.to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay=wdecay, learning_rate=lrate, 
                                                         betas=betas, eps=eps, amsgrad=amsgrad)
        if finetune_modules is not None:
            self.optimizer = self.model.freeze_backbone_optimizers(finetune_modules,weight_decay=wdecay, learning_rate=lrate, 
                                                         betas=betas, eps=eps, amsgrad=amsgrad)
            print('freezing backbone and now training only the finetune modules...')
            
        self.clip = clip
        self.regression_loss = missing_mse_loss
        self.classification_loss = torch.nn.BCEWithLogitsLoss()
        self.mclassification_loss = missing_ce_loss

        self.num_epochs = num_epochs
        
        self.result_path = result_path
        self.runid = runid
        self.device = device
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.multiclassification_weight = multiclassification_weight
        self.evaluate_metric = evaluate_metric
        self.skip_test_during_train = skip_test_during_train

        self.schedule_lr = schedule_lr
        if total_iters:
            self.total_iters = total_iters
        else:
            self.total_iters = num_epochs * steps_per_epoch

        self.lrate = lrate
        self.min_lrate = min_lrate
        self.warmup_iters = warmup_iters
        if lr_decay_iters is None:
            self.lr_decay_iters = self.total_iters
        else:
            self.lr_decay_iters = lr_decay_iters

    def train_epoch(self, train_loader, val_loader = None, test_loader = None, evaluate_epoch = 1):
        if self.evaluate_metric in ['rmse','mse','mae']:
            best_result = float('inf')  
        else:
            best_result = float('-inf') 
        pbar = tqdm(total=self.total_iters, desc='training')
        iter_num = 0
        val_str = ''
        test_str = ''
        with output(initial_len=11, interval=0) as output_lines:
            for epoch in range(1, self.num_epochs+1):
                running_reg_loss = 0
                running_cls_loss = 0
                running_mcls_loss = 0
                running_spectral_loss = 0 
                running_ortho_loss = 0 
                running_cluster_loss = 0 
                self.model.train()
                
                for data in train_loader:
                    if self.schedule_lr:
                        curr_lr_rate = self.get_lr(iter_num)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = curr_lr_rate
                    else:
                        curr_lr_rate = self.lrate

                    self.optimizer.zero_grad()

                    data = data.to(self.device)
                    reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, _ = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                    )
                    ## Loss compute
                    cls_loss = 0
                    mcls_loss = 0
                    reg_loss = 0

                    loss_val = torch.tensor(0.).to(self.device)
                    loss_val += sp_loss
                    loss_val += o_loss
                    loss_val += cl_loss
                    sp_loss = sp_loss.item()
                    o_loss = o_loss.item()
                    cl_loss = cl_loss.item()

                    if reg_pred is not None:
                        reg_pred = reg_pred.squeeze()
                        reg_y = data.reg_y.squeeze()
                        reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                        loss_val += reg_loss
                        reg_loss = reg_loss.item()

                    if cls_pred is not None:
                        cls_pred = cls_pred.squeeze()
                        cls_y = data.cls_y.squeeze()
                        cls_loss = self.classification_loss(cls_pred, cls_y) * self.classification_weight
                        loss_val += cls_loss
                        cls_loss = cls_loss.item()

                    if mcls_pred is not None:
                        mcls_y = data.mcls_y
                        mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
                        loss_val += mcls_loss
                        mcls_loss = mcls_loss.item()

                    loss_val.backward()

                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    self.model.temperature_clamp()                    
                    running_reg_loss += reg_loss
                    running_cls_loss += cls_loss
                    running_mcls_loss += mcls_loss

                    running_spectral_loss += sp_loss
                    running_ortho_loss += o_loss
                    running_cluster_loss += cl_loss
                    pbar.update(1)
                    iter_num += 1

                train_reg_loss = running_reg_loss / len(train_loader)
                train_cls_loss = running_cls_loss / len(train_loader)
                train_mcls_loss = running_mcls_loss / len(train_loader)
                train_spectral_loss = running_spectral_loss / len(train_loader)
                train_ortho_loss = running_ortho_loss / len(train_loader)
                train_cluster_loss = running_cluster_loss / len(train_loader)
                
                train_str1 = f"Train MSE Loss: {train_reg_loss:.4f}, Train CLS Loss: {train_cls_loss:.4f}, Train MCLS Loss: {train_mcls_loss:.4f}"
                train_str2 = f"Protein Cluster Region Loss - Ortho Loss: {train_ortho_loss:.4f}, Cluster Loss: {train_cluster_loss:.4f}"

                if epoch % evaluate_epoch == 0 and val_loader is not None:
                    val_result = self.eval(val_loader)
                    val_result = {k:round(v,4) for k, v in val_result.items()}
                    val_str =  f'Validation Results: ' + json.dumps(val_result, indent=4, sort_keys=True)
                    if self.evaluate_metric in ['rmse','mse','mae']:
                        if val_result[self.evaluate_metric] < best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False
                    else:
                        if val_result[self.evaluate_metric] > best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False

                    if better_than_previous:
                        best_result = val_result[self.evaluate_metric]
                        torch.save(self.model.state_dict(), os.path.join(self.result_path,'save_model_seed{}'.format(self.runid),'model.pt'))

                        if self.skip_test_during_train is False:
                            test_result = self.eval(test_loader)
                            test_result = {k:round(v,4) for k, v in test_result.items()}
                        else:
                            test_result = {}

                if epoch % evaluate_epoch == 0  and val_loader is None:
                    test_result = self.eval(test_loader)
                    
                    if self.evaluate_metric in ['rmse','mse','mae']:
                        if test_result[self.evaluate_metric] < best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False
                    else:
                        if test_result[self.evaluate_metric] > best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False

                    if better_than_previous:
                        torch.save(self.model.state_dict(), os.path.join(self.result_path,'save_model_seed{}'.format(self.runid),'model.pt'))
                
                test_result = {k:round(v,4) for k, v in test_result.items()}
                test_str = f'Test Results: ' + json.dumps(test_result, indent=4, sort_keys=True)
                output_lines[0] = ' '*30
                output_lines[1] = ' '*30
                output_lines[2] = '-'*40 
                output_lines[3] = f'Epoch {epoch:03d} with LR {curr_lr_rate:.6f}: Model Results'
                output_lines[4] = '-'*40 
                output_lines[5] = train_str1
                output_lines[6] = train_str2
                output_lines[7] = ' '*30
                output_lines[8] = val_str
                output_lines[9] = ' '*30
                output_lines[10] = test_str

                with open(self.result_path +'/full_result-{}.txt'.format(self.runid),'a+') as f:
                    f.write('-'*30 + f'\nEpoch: {epoch:03d} - Model Results\n' + '-'*30 + '\n')
                    f.write(train_str1 +'\n')
                    f.write(train_str2 +'\n')
                    f.write(val_str +'\n')
                    f.write(test_str +'\n')
                
    def train_step(self, train_loader, val_loader = None, test_loader = None, evaluate_step = 1):
        if self.evaluate_metric in ['rmse','mse','mae']:
            best_result = float('inf')  
        else:
            best_result = float('-inf') 
        running_reg_loss = 0
        running_cls_loss = 0
        running_mcls_loss = 0
        running_spectral_loss = 0 
        running_ortho_loss = 0 
        running_cluster_loss = 0 
        iter_num = 0
        pbar = tqdm(total=self.total_iters, desc='training')
        
        test_result = {}
        val_str = ''
        test_str = ''

        self.model.train()
        with output(initial_len=11, interval=0) as output_lines:
            while iter_num < self.total_iters:
                for data in train_loader:
                    # if iter_num <= 550000:
                    #     pbar.update(1)
                    #     iter_num += 1
                    #     continue
                    
                    if self.schedule_lr:
                        curr_lr_rate = self.get_lr(iter_num)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = curr_lr_rate
                    else:
                        curr_lr_rate = self.lrate

                    self.optimizer.zero_grad()
                    data = data.to(self.device)
                    reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, _ = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                    )
                    ## Loss compute
                    cls_loss = 0
                    mcls_loss = 0
                    reg_loss = 0

                    loss_val = torch.tensor(0.).to(self.device)
                    loss_val += sp_loss
                    loss_val += o_loss
                    loss_val += cl_loss
                    sp_loss = sp_loss.item()
                    o_loss = o_loss.item()
                    cl_loss = cl_loss.item()

                    if reg_pred is not None:
                        reg_pred = reg_pred.squeeze()
                        reg_y = data.reg_y.squeeze()
                        reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                        loss_val += reg_loss
                        reg_loss = reg_loss.item()

                    if cls_pred is not None:
                        cls_pred = cls_pred.squeeze()
                        cls_y = data.cls_y.squeeze()
                        cls_loss = self.classification_loss(cls_pred, cls_y) * self.classification_weight
                        loss_val += cls_loss
                        cls_loss = cls_loss.item()

                    if mcls_pred is not None:
                        mcls_y = data.mcls_y
                        mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
                        loss_val += mcls_loss
                        mcls_loss = mcls_loss.item()

                    loss_val.backward()

                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    self.model.temperature_clamp()

                    running_reg_loss += reg_loss
                    running_cls_loss += cls_loss
                    running_mcls_loss += mcls_loss
                    running_spectral_loss += sp_loss
                    running_ortho_loss += o_loss
                    running_cluster_loss += cl_loss
                    pbar.update(1)
                    iter_num += 1

                    if (iter_num % evaluate_step == 0) or (iter_num  == self.total_iters):
                        train_reg_loss = running_reg_loss / evaluate_step
                        train_cls_loss = running_cls_loss / evaluate_step
                        train_mcls_loss = running_mcls_loss / evaluate_step
                        train_spectral_loss = running_spectral_loss / evaluate_step
                        train_ortho_loss = running_ortho_loss / evaluate_step
                        train_cluster_loss = running_cluster_loss / evaluate_step
                        train_str1 = f"Train MSE Loss: {train_reg_loss:.4f}, Train CLS Loss: {train_cls_loss:.4f}, Train MCLS Loss: {train_mcls_loss:.4f}"
                        train_str2 = f"Protein Cluster Region Loss - Ortho Loss: {train_ortho_loss:.4f}, Cluster Loss: {train_cluster_loss:.4f}"

                        if val_loader is not None:
                            val_result = self.eval(val_loader)
                            val_result = {k:round(v,4) for k, v in val_result.items()}
                            val_str =  f'Validation Results: ' + json.dumps(val_result, indent=4, sort_keys=True)
                            
                            if self.evaluate_metric in ['rmse','mse','mae']:
                                if val_result[self.evaluate_metric] < best_result:
                                    better_than_previous = True
                                else:
                                    better_than_previous = False
                            else:
                                if val_result[self.evaluate_metric] > best_result:
                                    better_than_previous = True
                                else:
                                    better_than_previous = False

                            if better_than_previous:
                                best_result = val_result[self.evaluate_metric]
                                torch.save(self.model.state_dict(), os.path.join(self.result_path,'save_model_seed{}'.format(self.runid),'model.pt'))

                                if self.skip_test_during_train is False:
                                    test_result = self.eval(test_loader)
                                    test_result = {k:round(v,4) for k, v in test_result.items()}
                                else:
                                    test_result = {}
                        if val_loader is None:
                            test_result = self.eval(test_loader)
                            if self.evaluate_metric in ['rmse','mse','mae']:
                                if test_result[self.evaluate_metric] < best_result:
                                    better_than_previous = True
                                else:
                                    better_than_previous = False
                            else:
                                if test_result[self.evaluate_metric] > best_result:
                                    better_than_previous = True
                                else:
                                    better_than_previous = False
                            
                            if better_than_previous:
                                best_result = test_result[self.evaluate_metric]
                                torch.save(self.model.state_dict(), os.path.join(self.result_path,'save_model_seed{}'.format(self.runid),'model.pt'.format(iter_num)))
                            
                        test_result = {k:round(v,4) for k, v in test_result.items()}
                        test_str = f'Test Results: ' + json.dumps(test_result, indent=4, sort_keys=True)

                        output_lines[0] = ' '*30
                        output_lines[1] = ' '*30
                        output_lines[2] = '-'*40 
                        output_lines[3] = f'Training Step: {iter_num} with LR {curr_lr_rate:.6f}: Model Results'
                        output_lines[4] = '-'*40 
                        output_lines[5] = train_str1
                        output_lines[6] = train_str2
                        output_lines[7] = ' '*30
                        output_lines[8] = val_str
                        output_lines[9] = ' '*30
                        output_lines[10] = test_str

                        with open(self.result_path +'/full_result-{}.txt'.format(self.runid),'a+') as f:
                            f.write('-'*30 + f'\nTraining Step: {iter_num} - Model Results\n' + '-'*30 + '\n')
                            f.write(train_str1 +'\n')
                            f.write(train_str2 +'\n')
                            f.write(val_str +'\n')
                            f.write(test_str +'\n')
                        
                        ######## end evaluation and continue training ########
                        running_reg_loss = 0 
                        running_cls_loss = 0
                        running_mcls_loss = 0
                        running_spectral_loss = 0 
                        running_ortho_loss = 0 
                        running_cluster_loss = 0 
                        self.model.train()
                        ######## end evaluation and continue training ########
                    if iter_num == self.total_iters: break

            pbar.close()

            
    def eval(self, data_loader):
        reg_preds = []
        reg_truths = []
        cls_preds = []
        cls_truths = []

        mcls_preds = []
        mcls_truths = []

        running_reg_loss = 0
        running_cls_loss = 0
        running_mcls_loss = 0

        running_spectral_loss = 0 
        running_ortho_loss = 0 
        running_cluster_loss = 0  

        self.model.eval()
        eval_result = {}
        with torch.no_grad():
            for data in tqdm(data_loader, leave=False, desc='evaluating'):
                data = data.to(self.device)
                reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, _ = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                    )
                ## Loss compute
                cls_loss = 0
                mcls_loss = 0
                reg_loss = 0

                loss_val = 0
                loss_val += sp_loss
                loss_val += o_loss
                loss_val += cl_loss
                sp_loss = sp_loss.item()
                o_loss = o_loss.item()
                cl_loss = cl_loss.item()

                if reg_pred is not None:
                    reg_pred = reg_pred.squeeze().reshape(-1)
                    reg_y = data.reg_y.squeeze().reshape(-1)
                    reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                    loss_val += reg_loss
                    reg_loss = reg_loss.item()
                    reg_preds.append(reg_pred)
                    reg_truths.append(reg_y)

                if cls_pred is not None:
                    cls_pred = cls_pred.squeeze().reshape(-1)
                    cls_y = data.cls_y.squeeze().reshape(-1)
                    cls_loss = self.classification_loss(cls_pred, cls_y) * self.classification_weight
                    loss_val += cls_loss
                    cls_loss = cls_loss.item()
                    cls_preds.append(cls_pred)
                    cls_truths.append(cls_y)

                if mcls_pred is not None:
                    mcls_y = data.mcls_y
                    mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
                    loss_val += mcls_loss
                    mcls_loss = mcls_loss.item()
                    mcls_preds.append(mcls_pred)
                    mcls_truths.append(mcls_y)

                running_reg_loss += reg_loss
                running_cls_loss += cls_loss
                running_mcls_loss += mcls_loss
                running_spectral_loss += sp_loss
                running_ortho_loss += o_loss
                running_cluster_loss += cl_loss

            eval_reg_loss = running_reg_loss / len(data_loader)
            eval_cls_loss = running_cls_loss / len(data_loader)
            eval_mcls_loss = running_mcls_loss / len(data_loader)
            eval_spectral_loss = running_spectral_loss / len(data_loader)
            eval_ortho_loss = running_ortho_loss / len(data_loader)
            eval_cluster_loss = running_cluster_loss / len(data_loader)

            eval_result['regression_loss'] = eval_reg_loss
            eval_result['classification_loss'] = eval_cls_loss
            eval_result['multiclassification_loss'] = eval_mcls_loss
            eval_result['spectral_loss'] = eval_spectral_loss
            eval_result['ortho_loss'] = eval_ortho_loss
            eval_result['cluster_loss'] = eval_cluster_loss

        if len(reg_truths) > 0:
            reg_preds = torch.cat(reg_preds).detach().cpu().numpy()
            reg_truths = torch.cat(reg_truths).detach().cpu().numpy()
            eval_reg_result = evaluate_reg(reg_truths, reg_preds)
            eval_result.update(eval_reg_result)

        if len(cls_truths) > 0:
            cls_preds = torch.sigmoid(torch.cat(cls_preds)).detach().cpu().numpy()
            cls_truths = torch.cat(cls_truths).detach().cpu().numpy()
        
            eval_cls_result = evaluate_cls(cls_truths, cls_preds, threshold=0.5)
            eval_result.update(eval_cls_result)

        if len(mcls_truths) > 0:
            mcls_preds = torch.softmax(torch.cat(mcls_preds),dim=-1)
            mcls_truths = torch.cat(mcls_truths)
            mask = ~torch.isnan(mcls_truths)
            if mask.sum() > 0:
                mcls_truths = mcls_truths[mask]
                mcls_truths = mcls_truths.long().detach().cpu().numpy()
                mcls_preds = mcls_preds[mask]
                mcls_preds = mcls_preds.detach().cpu().numpy()
                eval_mcls_result = evaluate_mcls(mcls_truths, mcls_preds)
                eval_result.update(eval_mcls_result)

        return eval_result

    def pred(self, model, data_loader, store_interpret=True):
        reg_preds = []
        reg_truths = []
        cls_preds = []
        cls_truths = []
        mcls_preds = []
        mcls_truths = []

        running_reg_loss = 0
        running_cls_loss = 0
        running_mcls_loss = 0
        reg_tuples = None
        cls_tuples = None

        running_spectral_loss = 0 
        running_ortho_loss = 0 
        running_cluster_loss = 0  


        eval_result = {}
        interpretation_result = {}

        model.eval()
        
        with torch.no_grad():
            for data in tqdm(data_loader):
                data = data.to(self.device)
                reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                )
                ## Loss compute
                cls_loss = 0
                reg_loss = 0
                mcls_loss = 0

                loss_val = 0
                loss_val += sp_loss
                loss_val += o_loss
                loss_val += cl_loss
                sp_loss = sp_loss.item()
                o_loss = o_loss.item()
                cl_loss = cl_loss.item()

                if reg_pred is not None:
                    reg_pred = reg_pred.squeeze().reshape(-1)
                    reg_y = data.reg_y.squeeze().reshape(-1)
                    reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                    loss_val += reg_loss
                    reg_loss = reg_loss.item()
                    reg_preds.append(reg_pred)
                    reg_truths.append(reg_y)
                    reg_tuples = list(zip(reg_y, reg_pred))
                    

                if cls_pred is not None:
                    cls_pred = cls_pred.squeeze().reshape(-1)
                    cls_y = data.cls_y.squeeze().reshape(-1)
                    cls_loss = self.classification_loss(cls_pred, cls_y) * self.classification_weight
                    loss_val += cls_loss
                    cls_loss = cls_loss.item()
                    cls_preds.append(cls_pred)
                    cls_truths.append(cls_y)
                    cls_tuples = list(zip(cls_y, cls_pred))

                if mcls_pred is not None:
                    mcls_y = data.mcls_y
                    mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
                    loss_val += mcls_loss
                    mcls_loss = mcls_loss.item()
                    mcls_preds.append(mcls_pred)
                    mcls_truths.append(mcls_y)


                interaction_key = list(zip(data.mol_key, data.prot_key))
                if store_interpret:
                    attention_dict = store_attention_result(attention_dict, interaction_key, reg_tuples, cls_tuples)
                    interpretation_result.update(attention_dict)

                running_reg_loss += reg_loss
                running_cls_loss += cls_loss
                running_mcls_loss += mcls_loss
                running_spectral_loss += sp_loss
                running_ortho_loss += o_loss
                running_cluster_loss += cl_loss

            eval_reg_loss = running_reg_loss / len(data_loader)
            eval_cls_loss = running_cls_loss / len(data_loader)
            eval_mcls_loss = running_mcls_loss / len(data_loader)
            eval_spectral_loss = running_spectral_loss / len(data_loader)
            eval_ortho_loss = running_ortho_loss / len(data_loader)
            eval_cluster_loss = running_cluster_loss / len(data_loader)

            eval_result['regression_loss'] = eval_reg_loss
            eval_result['classification_loss'] = eval_cls_loss
            eval_result['multiclassification_loss'] = eval_mcls_loss
            eval_result['spectral_loss'] = eval_spectral_loss
            eval_result['ortho_loss'] = eval_ortho_loss
            eval_result['cluster_loss'] = eval_cluster_loss

        if len(reg_truths) > 0:
            reg_preds = torch.cat(reg_preds).detach().cpu().numpy()
            reg_truths = torch.cat(reg_truths).detach().cpu().numpy()
            try:
                eval_reg_result = evaluate_reg(reg_truths, reg_preds)
            except:
                eval_reg_result = {}
            eval_result.update(eval_reg_result)

        if len(cls_truths) > 0:
            cls_preds = torch.sigmoid(torch.cat(cls_preds)).detach().cpu().numpy()
            cls_truths = torch.cat(cls_truths).detach().cpu().numpy()
            try:
                eval_cls_result = evaluate_cls(cls_truths, cls_preds, threshold=0.5)
            except: 
                eval_cls_result = {}
            eval_result.update(eval_cls_result)
        
        if len(mcls_truths) > 0:
            mcls_preds = torch.softmax(torch.cat(mcls_preds),dim=-1)
            mcls_truths = torch.cat(mcls_truths)
            mask = ~torch.isnan(mcls_truths)
            if mask.sum() > 0:
                mcls_truths = mcls_truths[mask]
                mcls_truths = mcls_truths.long().detach().cpu().numpy()
                mcls_preds = mcls_preds[mask]
                mcls_preds = mcls_preds.detach().cpu().numpy()
                try:
                    eval_mcls_result = evaluate_mcls(mcls_truths, mcls_preds)
                except:
                    eval_mcls_result = {}
                eval_result.update(eval_mcls_result)

        final_dict = {
            'regression_prediction':reg_preds,
            'regression_truth':reg_truths,
            'classification_prediction':cls_preds,
            'classification_truth':cls_truths,
            'mclassification_prediction':mcls_preds,
            'mclassification_truth':mcls_truths,
            'evaluation_result':eval_result,
            'interpretation_result':interpretation_result
        }

        return final_dict

    def get_lr(self,iter):
        # 1) linear warmup for warmup_iters steps
        if iter < self.warmup_iters:
            return self.lrate * iter / self.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > self.lr_decay_iters:
            return self.min_lrate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        
        return self.min_lrate + coeff * (self.lrate - self.min_lrate)


def masked_mse_loss(pred, true):
    mask = ~torch.isnan(true)
    pred = torch.masked_select(pred, mask)
    true = torch.masked_select(true, mask)
    mse_val =  torch.mean( (true-pred)**2 )

    return mse_val

def store_attention_result(attention_dict, keys, reg_tuples=None, cls_tuples=None):
    interpret_dict = {}

    unbatched_residue_score = unbatch(attention_dict['residue_final_score'],attention_dict['protein_residue_index'])
    unbatched_atom_score = unbatch(attention_dict['atom_final_score'], attention_dict['drug_atom_index'])

    unbatched_residue_layer_score = unbatch(attention_dict['residue_layer_scores'],attention_dict['protein_residue_index'])
    unbatched_clique_layer_score = unbatch(attention_dict['clique_layer_scores'], attention_dict['drug_clique_index'])

    for idx, key in enumerate(keys):
        interpret_dict[key] = {
            'residue_score': unbatched_residue_score[idx].detach().cpu().numpy(),
            'atom_score':unbatched_atom_score[idx].detach().cpu().numpy(),
            'residue_layer':unbatched_residue_layer_score[idx].detach().cpu().numpy(),
            'clique_layer':unbatched_clique_layer_score[idx].detach().cpu().numpy(),
            'mol_feature':attention_dict['mol_feature'][idx].detach().cpu().numpy(),
            'prot_feature':attention_dict['prot_feature'][idx].detach().cpu().numpy(),
            'interaction_fingerprint':attention_dict['interaction_fingerprint'][idx].detach().cpu().numpy(),
        }
        if cls_tuples:
            interpret_dict[key]['classification_truth'] = cls_tuples[idx][0].item()
            interpret_dict[key]['classification_prediction'] = cls_tuples[idx][1].item()
        if reg_tuples:
            interpret_dict[key]['regression_truth'] = reg_tuples[idx][0].item()
            interpret_dict[key]['regression_prediction'] = reg_tuples[idx][1].item()
    
    return interpret_dict


def missing_mse_loss(pred, true, threshold=5.000):
    loss = torch.tensor(0.).to(pred.device)
    
    ## true labels available
    if (~torch.isnan(true)).any():
        real_mask = ~torch.isnan(true)
        real_pred = torch.masked_select(pred, real_mask)
        real_true = torch.masked_select(true, real_mask)
        loss += torch.mean((real_true-real_pred)**2 )

    # ## missing labels
    # if torch.isnan(true).any():
    #     miss_mask = torch.isnan(true)
    #     miss_pred = torch.masked_select(pred, miss_mask)
    #     miss_diff = (miss_pred - threshold).relu()
    #     loss += torch.mean(miss_diff**2)

    return loss

def missing_ce_loss(pred, true, negative_cls=1):
    mclass_criterion = torch.nn.CrossEntropyLoss()
    negative_class_criterion = torch.nn.BCELoss()
    loss = torch.tensor(0.).to(pred.device)
    counter = 0

    if (~torch.isnan(true)).any():
        real_mask = ~torch.isnan(true)
        real_pred = pred[real_mask]
        real_true = true[real_mask]

        ## unknown
        unknown_mask = torch.where(real_true == 1000)[0]
        unknown_pred = real_pred[unknown_mask]
        if len(unknown_pred) > 0:
            unknown_pred = unknown_pred.softmax(dim=-1)
            ## take binder class (agonist and antagonist) only ##
            positive_cls = torch.ones(unknown_pred.shape[-1]).bool()
            positive_cls[negative_cls]=False
            positive_cls = positive_cls.to(pred.device)
            unknown_pred = unknown_pred[:,positive_cls].sum(dim=-1)
            ## take binder class (agonist and antagonist) only ##
            unknown_true = torch.ones(unknown_pred.size(0)).float().to(pred.device) ## all of them are positives
            unknown_pred = torch.where(torch.isnan(unknown_pred), torch.zeros_like(unknown_pred), unknown_pred).clamp(0,1)

            loss += negative_class_criterion(unknown_pred, unknown_true)
            counter += 1

        ## known values
        known_mask = torch.where(real_true != 1000)[0]
        known_pred = real_pred[known_mask]
        known_true = real_true[known_mask]
        if len(known_pred) > 0: 
            known_true = known_true.long()
            loss += mclass_criterion(known_pred, known_true)
            counter += 1

    return loss