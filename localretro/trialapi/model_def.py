from typing import Any, Dict, Union, Sequence
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler, MetricReducer

#NEW - import
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from utils import mkdir_p, collate_molgraphs, collate_molgraphs_test
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, EarlyStopping
from dataset import USPTODataset, USPTOTestDataset
from functools import partial
from dgl.data.utils import Subset
from models import LocalRetro

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

from collections.abc import Iterable

class MyAvgMetricReducer(MetricReducer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.counts = 0

    # User-defined mechanism for collecting values throughout
    # training or validation. This update() mechanism demonstrates
    # a computationally- and memory-efficient way to store the values.
    def update(self, value):
        if not (isinstance(value, Iterable)):
            value = [value]
        self.sum += sum(value)
        self.counts += 1

    def per_slot_reduce(self):
        # Because the chosen update() mechanism is so
        # efficient, this is basically a noop.
        return self.sum, self.counts

    def cross_slot_reduce(self, per_slot_metrics):
        # per_slot_metrics is a list of (sum, counts) tuples
        # returned by the self.pre_slot_reduce() on each slot
        sums, counts = zip(*per_slot_metrics)
        return sum(sums) / sum(counts) if sum(counts) else 0

class LocalRetroPytorch(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        # NEW - Custom Metric Reducer
        self.custom_loss = context.wrap_reducer(
            MyAvgMetricReducer(), name="custom_loss"
        )

        self.context = context
        
        self.model_name = 'LocalRetro_%s.pth' % self.context.get_hparam("dataset")
        self.model_path = './models/' + self.model_name
        self.data_dir = './data/%s' % self.context.get_hparam("dataset")
        mkdir_p('./models')  

        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
        self.node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

        AtomTemplate_n = len(pd.read_csv('%s/atom_templates.csv' % self.data_dir))
        BondTemplate_n = len(pd.read_csv('%s/bond_templates.csv' % self.data_dir))
        self.AtomTemplate_n = AtomTemplate_n
        self.BondTemplate_n = BondTemplate_n
        self.in_node_feats = self.node_featurizer.feat_size()
        self.in_edge_feats = self.edge_featurizer.feat_size()
        
        self.node_out_feats = self.context.get_hparam("node_out_feats")
        self.edge_hidden_feats = self.context.get_hparam("edge_hidden_feats")
        self.num_step_message_passing = self.context.get_hparam("num_step_message_passing")
        self.attention_heads = self.context.get_hparam("attention_heads")
        self.attention_layers = self.context.get_hparam("attention_layers")
        self.drop_out = self.context.get_hparam("drop_out")

        self.args = {'dataset': self.context.get_hparam("dataset"), 'data_dir': self.data_dir}

        model = LocalRetro(
            drop_out=self.drop_out,           
            node_in_feats=self.in_node_feats,
            edge_in_feats=self.in_edge_feats,
            node_out_feats=self.node_out_feats,
            edge_hidden_feats=self.edge_hidden_feats,
            num_step_message_passing=self.num_step_message_passing,
            attention_heads=self.attention_heads,
            attention_layers=self.attention_layers,
            AtomTemplate_n=self.AtomTemplate_n,
            BondTemplate_n=self.BondTemplate_n)

        loss_criterion = nn.CrossEntropyLoss(reduction = 'none')
        optimizer = Adam(model.parameters(), lr=self.context.get_hparam("learning_rate"), weight_decay=self.context.get_hparam("weight_decay"))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.context.get_hparam("schedule_step"))      
        stopper = EarlyStopping(mode = 'lower', patience=self.context.get_hparam("patience"), filename=self.model_path)

        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(optimizer)
        self.lr_scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH) 
        self.loss_criterion = loss_criterion
        self.stopper = stopper

        dataset = USPTODataset(self.args, 
                            smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=self.node_featurizer,
                            edge_featurizer=self.edge_featurizer)
        
        self.train_set, self.val_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids)

    def build_training_data_loader(self) -> DataLoader:
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.context.get_per_slot_batch_size(), shuffle=True,
                                  collate_fn=collate_molgraphs, num_workers=0)
        return train_loader

    def build_validation_data_loader(self) -> DataLoader:
        val_loader = DataLoader(dataset=self.val_set, batch_size=self.context.get_per_slot_batch_size(),
                                  collate_fn=collate_molgraphs, num_workers=0)
        return val_loader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
        self.model.train()
        smiles, bg, atom_labels, bond_labels = batch
        if len(smiles) == 1:
            exit()

        atom_logits, bond_logits, _ = predict(self.model, bg)

        loss_a = self.loss_criterion(atom_logits, atom_labels)
        loss_b = self.loss_criterion(bond_logits, bond_labels)
        total_loss = torch.cat([loss_a, loss_b]).mean()

        self.context.backward(total_loss) 
        self.grad_clip_fn = lambda params: torch.nn.utils.clip_grad_norm_(params, self.context.get_hparam("max_clip"))
        self.context.step_optimizer(self.optimizer, self.grad_clip_fn)
        self.lr_scheduler.step()

        # NEW - Custom Metric Reducer
        self.custom_loss.update(total_loss.item())
        train_loss, batch_id = self.custom_loss.per_slot_reduce()
        if batch_id % 100 == 0:
            print('Batch: %d, train_loss: %.4f' % (batch_id, train_loss))

        if self.context.is_epoch_end():
            self.custom_loss.reset()
            print('Average train loss per epoch: %.4f' % (train_loss / batch_id))

        return {"total_loss": total_loss.item(), "avg_train_loss": train_loss / batch_id}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        self.model.eval()
        smiles, bg, atom_labels, bond_labels = batch
        atom_logits, bond_logits, _ = predict(self.model, bg)
        
        loss_a = self.loss_criterion(atom_logits, atom_labels)
        loss_b = self.loss_criterion(bond_logits, bond_labels)
        total_loss = torch.cat([loss_a, loss_b]).mean()

        # NEW - Custom Metric Reducer
        self.custom_loss.update(total_loss.item())
        val_loss, batch_id = self.custom_loss.per_slot_reduce()
        if batch_id % 100 == 0:
            print('Batch: %d, val_loss: %.4f' % (batch_id, val_loss))

        if self.context.is_epoch_end():
            self.custom_loss.reset()
            print('Average val loss per epoch: %.4f' % (val_loss / batch_id))

        return {"val_loss": total_loss.item(), "avg_val_loss": (val_loss / batch_id)}
  
def predict(model, bg):
    node_feats = bg.ndata.pop('h')
    edge_feats = bg.edata.pop('e')
    return model(bg, node_feats, edge_feats)