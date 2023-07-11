from argparse import ArgumentParser

import torch
import sklearn
import torch.nn as nn

from utils import init_featurizer, mkdir_p, get_configure, load_model, load_dataloader, predict

import determined as det
import pathlib

# NEW: Import torch distributed libraries.
import torch.distributed as dist

def run_a_train_epoch(args, core_context, epoch, model, data_loader, loss_criterion, optimizer, steps_completed, op):
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

        # NEW: Report metrics only on rank 0: only the chief worker
        # may report training metrics and progress, or upload checkpoints.
        if core_context.distributed.rank == 0:      
            if batch_id % args['print_every'] == 0:
                print('\repoch %d/%d, batch %d/%d, loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), total_loss), end='', flush=True)

    # NEW: Report metrics only on rank 0: only the chief worker
    # may report training metrics and progress, or upload checkpoints.
    if core_context.distributed.rank == 0:
        core_context.train.report_training_metrics(
                steps_completed=steps_completed,
                metrics={"train_loss": train_loss/len(data_loader)},
            )
    # NEW: Report metrics only on rank 0: only the chief worker
    # may report training metrics and progress, or upload checkpoints.
    if core_context.distributed.rank == 0:
        op.report_progress(epoch)
    # NEW
        print('\nepoch %d/%d, training loss: %.4f' % (epoch + 1, args['num_epochs'], train_loss/batch_id))

def run_an_eval_epoch(args, core_context, model, data_loader, loss_criterion, steps_completed, op):
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

        # NEW: Report metrics only on rank 0: only the chief worker
        # may report training metrics and progress, or upload checkpoints.
        if core_context.distributed.rank == 0:
            core_context.train.report_validation_metrics(
                steps_completed=steps_completed,
                metrics={"val_loss": val_loss/len(data_loader)},   
            )

    return val_loss/batch_id

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
    # NEW moved and changed to get device from local_rank
    args['device'] = torch.device(core_context.distributed.local_rank) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])

    model_name = 'LocalRetro_%s.pth' % args['dataset']
    args['model_path'] = './models/' + model_name
    args['config_path'] = './data/configs/%s' % args['config']
    args['data_dir'] = './data/%s' % args['dataset']
    mkdir_p('./models')                          
    args = init_featurizer(args)

    info = det.get_cluster_info()
    assert info is not None, "this example only runs on-cluster"

    hparams = info.trial.hparams

    model, loss_criterion, optimizer, scheduler, stopper = load_model(args, hparams, device=args['device']) 
    # New - add corecontext & hparams
    train_loader, val_loader, test_loader = load_dataloader(args, core_context, hparams)
    
    latest_checkpoint = info.latest_checkpoint
    if latest_checkpoint is None:
        epochs_completed = 0
    else:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            model, epochs_completed = load_state(model, path, info.trial.trial_id)

    epoch = epochs_completed
    last_checkpoint_batch = None

    for op in core_context.searcher.operations():
        while epoch < op.length:
            epochs_completed = epoch + 1
            steps_completed = epoch * len(train_loader)

            run_a_train_epoch(args, core_context, epoch, model, train_loader, loss_criterion, optimizer, steps_completed, op)
            val_loss = run_an_eval_epoch(args, core_context, model, val_loader, loss_criterion, steps_completed, op)
            early_stop = stopper.step(val_loss, model) 
            scheduler.step()

            checkpoint_metadata_dict = {"steps_completed": steps_completed}

            epoch += 1

            # NEW: Report metrics only on rank 0: only the chief worker
            # may report training metrics and progress, or upload checkpoints.
            # Store checkpoints only on rank 0.
            if core_context.distributed.rank == 0:
                with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
                    torch.save(model.state_dict(), path / "checkpoint.pt")
                    with path.joinpath("state").open("w") as f:
                        f.write(f"{epochs_completed},{info.trial.trial_id}")

            if core_context.preempt.should_preempt():
                return

            if core_context.distributed.rank == 0:
                print('epoch %d/%d, validation loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
                print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))
                if early_stop:
                    print ('Early stopped!!')
                    break
        # NEW: Report metrics only on rank 0: only the chief worker
        # may report training metrics and progress, or upload checkpoints.
        if core_context.distributed.rank == 0:
            op.report_completed(val_loss)

    stopper.load_checkpoint(model)
    test_loss = run_an_test_epoch(args, model, test_loader, loss_criterion)
    # NEW: Report metrics only on rank 0: only the chief worker
    # may report training metrics and progress, or upload checkpoints.
    if core_context.distributed.rank == 0:
        print('test loss: %.4f' % test_loss)

def load_state(model, checkpoint_directory, trial_id):
    checkpoint_directory = pathlib.Path(checkpoint_directory)

    with checkpoint_directory.joinpath("checkpoint.pt").open("rb") as f:
        model.load_state_dict(torch.load(f))
    with checkpoint_directory.joinpath("state").open("r") as f:
        epochs_completed, ckpt_trial_id = [int(field) for field in f.read().split(",")]
    
    if ckpt_trial_id != trial_id:
        epochs_completed = 0

    return model, epochs_completed

if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=5, help='Maximum number of epochs for training')  #NEW default 50 to 20
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=100, help='Print the training progress every X mini-batches') #NEW default 20 -> 100
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    
    # NEW - Move args['device'] to main()
    
    # NEW: Initialize process group using torch.
    dist.init_process_group("nccl")

    # NEW: Initialize distributed context using from_torch_distributed
    # (obtains info such as rank, size, etc. from default torch
    # environment variables).
    distributed = det.core.DistributedContext.from_torch_distributed()

    with det.core.init(distributed=distributed) as core_context:
        main(args, core_context=core_context)
