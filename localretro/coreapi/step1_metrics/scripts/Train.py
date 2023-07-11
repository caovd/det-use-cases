from argparse import ArgumentParser

import torch
import sklearn
import torch.nn as nn

from utils import init_featurizer, mkdir_p, get_configure, load_model, load_dataloader, predict

import determined as det

def run_a_train_epoch(args, core_context, epoch, model, data_loader, loss_criterion, optimizer, steps_completed):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, atom_labels, bond_labels = batch_data
        if len(smiles) == 1:
            continue
           
        atom_labels, bond_labels = atom_labels.to(args['device']), bond_labels.to(args['device'])
        atom_logits, bond_logits, _ = predict(args, model, bg)

        loss_a = loss_criterion(atom_logits, atom_labels)
        loss_b = loss_criterion(bond_logits, bond_labels)
        total_loss = torch.cat([loss_a, loss_b]).mean()
        train_loss += total_loss.item()
        
        optimizer.zero_grad()      
        total_loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
                
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), total_loss), end='', flush=True)

        # NEW: Print training progress and loss at specified intervals
        # starting from the first batch.
        # batches_completed = batch_id + 1
        # Docs snippet start: report training metrics for each batch
        # NEW: Report training metrics to Determined
        # master via core_context.
        # Index by (batch_idx + 1) * (epoch-1) * len(train_loader)
        # to continuously plot loss on one graph for consecutive
        # epochs.
        # core_context.train.report_training_metrics(
        #     steps_completed=batches_completed + epoch * len(data_loader),
        #     metrics={"train_loss": train_loss/batch_id if batch_id else train_loss},
        # )
        # Docs snippet end: report training metrics 

    # Docs snippet start: report training metrics for each epoch
    core_context.train.report_training_metrics(
            steps_completed=steps_completed,
            metrics={"train_loss": train_loss/len(data_loader)},
        )
    # Docs snippet end: report training metrics for each epoch

    print('\nepoch %d/%d, training loss: %.4f' % (epoch + 1, args['num_epochs'], train_loss/batch_id))

# Docs snippet start:
# NEW: Modify function header to include 
# core_context for metric reporting and a steps_completed parameter to
# plot metrics.
def run_an_eval_epoch(args, core_context, model, data_loader, loss_criterion, steps_completed):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, atom_labels, bond_labels = batch_data
            atom_labels, bond_labels = atom_labels.to(args['device']), bond_labels.to(args['device'])
            atom_logits, bond_logits, _ = predict(args, model, bg)
            
            loss_a = loss_criterion(atom_logits, atom_labels)
            loss_b = loss_criterion(bond_logits, bond_labels)
            total_loss = torch.cat([loss_a, loss_b]).mean()
            val_loss += total_loss.item()

        # Docs snippet end: include args
        # Docs snippet start: report validation metrics
        # NEW: Report validation metrics to Determined master
        # via core_context.
        core_context.train.report_validation_metrics(
            steps_completed=steps_completed,
            metrics={"val_loss": val_loss/len(data_loader)},   
        )
        # Docs snippet end: report validation metrics

    return val_loss/batch_id

# NEW create a separate function for testing to avoid duplicate Determined AI report of val loss
def run_an_test_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, atom_labels, bond_labels = batch_data
            atom_labels, bond_labels = atom_labels.to(args['device']), bond_labels.to(args['device'])
            atom_logits, bond_logits, _ = predict(args, model, bg)
            
            loss_a = loss_criterion(atom_logits, atom_labels)
            loss_b = loss_criterion(bond_logits, bond_labels)
            total_loss = torch.cat([loss_a, loss_b]).mean()
            test_loss += total_loss.item()

    return test_loss/batch_id 

def main(args, core_context):
    model_name = 'LocalRetro_%s.pth' % args['dataset']
    # NEW: Modify path
    args['model_path'] = './models/' + model_name
    args['config_path'] = './data/configs/%s' % args['config']
    args['data_dir'] = './data/%s' % args['dataset']
    mkdir_p('./models')                          
    args = init_featurizer(args)
    model, loss_criterion, optimizer, scheduler, stopper = load_model(args)   
    train_loader, val_loader, test_loader = load_dataloader(args)
    for epoch in range(args['num_epochs']):

        # Docs snippet start: calculate steps completed
        # NEW: Calculate steps_completed for plotting test metrics.
        steps_completed = epoch * len(train_loader)
        # Docs snippet end: calculate steps completed

        run_a_train_epoch(args, core_context, epoch, model, train_loader, loss_criterion, optimizer, steps_completed=steps_completed)
        val_loss = run_an_eval_epoch(args, core_context, model, val_loader, loss_criterion, steps_completed=steps_completed)
        early_stop = stopper.step(val_loss, model) 
        scheduler.step()
        print('epoch %d/%d, validation loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
        print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))
        if early_stop:
            print ('Early stopped!!')
            break

    stopper.load_checkpoint(model)
    test_loss = run_an_test_epoch(args, model, test_loader, loss_criterion)
    print('test loss: %.4f' % test_loss)
    
if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=20, help='Maximum number of epochs for training')  #NEW default 50 to 20
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=100, help='Print the training progress every X mini-batches') #NEW default 20 -> 100
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])

    # NEW: Establish new determined.core.Context and pass to main
    # function.
    with det.core.init() as core_context:
        main(args, core_context=core_context)
