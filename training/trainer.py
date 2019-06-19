import sys
import time
import numpy as np
import torch
import warnings
from data_models.evaluation_result import EvaluationResult
from data_models.objective_func import ObjectiveFunction
from datasets.radar_dataset import DatasetPartition, DataContent
from training.early_stopping import EarlyStopping
from training.rd_evaluation import evaluate_rd
from run_scripts import device, task_id, JOB_DIR, tensorboard_writer
from run_scripts import verbose, visualize, tensorboard_logging, memory_logging, RESIDUAL_LEARNING, print_, \
    is_cluster_run
from training.rd_log_mag_evaluation import evaluate_rd_log_mag
from utils.loading import data_loader_for_dataset
from utils.mem_usage import print_torch_mem_usage
from utils.plotting import plot_losses
from utils.printing import PrintColors


def train_with_hyperparameter_config(dataset, hyperparameters, task, is_classification=False, shuffle_data=True):
    optimization_algo = hyperparameters.optimization_algo
    criterion = hyperparameters.criterion
    scheduler_partial = hyperparameters.scheduler_partial
    batch_size = hyperparameters.batch_size
    learning_rate = hyperparameters.learning_rate
    num_epochs = hyperparameters.num_epochs
    num_model_initializations = hyperparameters.num_model_initializations
    output_size = hyperparameters.output_size
    model = hyperparameters.model

    if is_classification:
        prediction_size = 1
    else:
        prediction_size = output_size

    if verbose:
        print_('Running task: {}'.format(task))
        print_('Max Epochs: {}'.format(num_epochs))
        print_()
        print_(hyperparameters)
        print_()
        print_('# Training #')

    train_loader = data_loader_for_dataset(dataset, batch_size=batch_size, shuffle=shuffle_data)
    val_loader = data_loader_for_dataset(dataset.clone_for_new_active_partition(DatasetPartition.VALIDATION), batch_size=batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}
    count_batch_iterations = {'train': int(dataset_sizes['train'] / batch_size), 'val': int(dataset_sizes['val'] / batch_size)}

    try:
        criterion.weight = criterion.weight.to(device)
    except AttributeError:
        pass

    since = time.time()

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_model_state_dict = None
    last_acc = {'train': None, 'val': None}
    use_validation = dataset_sizes['val'] > 0
    stopping_strategy = EarlyStopping(steps_to_wait=50)
    model = model.to(device)

    # # # # # train and evaluate the configuration # # # # #
    for model_initialization in range(num_model_initializations):
        model.reset()

        try:
            if tensorboard_logging:
                model.set_tensorboardx_logging_active(tensorboard_logging)
        except AttributeError:
            warnings.warn('Model does not support tensorboard logging!')
            pass

        optimizer = optimization_algo(params=model.parameters(), lr=learning_rate)

        if verbose:
            print_('Model initialization {}/{}'.format(model_initialization + 1, num_model_initializations))
            line = construct_formatted_values_headline()
            print_(line)

        losses = {'train': np.zeros(num_epochs), 'val': np.zeros(num_epochs)}
        if scheduler_partial is not None:
            scheduler = scheduler_partial(optimizer)
        batch_iteration = {'train': 0, 'val': 0}
        for epoch in range(num_epochs):

            epoch_start_time = time.time()

            try:
                scheduler.step()
            except UnboundLocalError:
                pass
            if tensorboard_logging:
                for param_group in optimizer.param_groups:
                    tensorboard_writer.add_scalar('data/learning_rate', param_group['lr'], epoch)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                curr_count_sample = 0

                mem_usage = print_torch_mem_usage(None, print_mem=False)

                for inputs_batch, targets_batch, filter_mask_batch, object_masks, noise_masks in dataloaders[phase]:

                    inputs_for_training = inputs_batch[filter_mask_batch].to(device)
                    targets_for_learning = targets_batch[filter_mask_batch].to(device)

                    sample_batch_size = inputs_for_training.shape[0]

                    if sample_batch_size <= 1:  # sample_batch_size == 1: does not work with batch_norm
                        warnings.warn('Skipping batch with size {}. Batch norm requires a batch size >= 2.'.format(sample_batch_size))
                        continue

                    try:
                        model.init_hidden_state()  # required by LSTM
                    except AttributeError:
                        pass

                    # track history only if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs_for_training)

                        if RESIDUAL_LEARNING:
                            targets_for_loss = inputs_for_training - targets_for_learning
                        else:
                            targets_for_loss = targets_for_learning

                        loss = criterion(outputs, targets_for_loss, object_masks, noise_masks)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            loss.backward()
                            optimizer.step()

                    running_loss += ObjectiveFunction.loss_to_running_loss(loss.item(), sample_batch_size)

                    if is_classification:
                        _, class_predictions = torch.max(outputs.detach(), 1)
                        running_corrects += torch.sum(class_predictions == targets_for_learning.detach()).double()

                    if tensorboard_logging:
                        tensorboard_writer.add_scalar('data/{}_batch_loss'.format(phase), loss.item(),
                                                      batch_iteration[phase])
                        if phase == 'train' and batch_iteration[phase] == 0:
                            try:
                                tensorboard_writer.add_graph(model, inputs_for_training)
                            except AssertionError:
                                warnings.warn('Model does not support tensorboard graphs!')
                    curr_count_sample += sample_batch_size
                    current_loss = ObjectiveFunction.loss_from_running_loss(running_loss, curr_count_sample)

                    if verbose and not is_cluster_run and batch_iteration[phase] % max(int(count_batch_iterations[phase]/ 100), 1) == 0:
                        if phase == 'train':
                            line = '\r' + construct_formatted_values_line(epoch, current_loss, best_train_loss,
                                                                          None, best_val_loss, time.time() - epoch_start_time,
                                                                          is_classification, running_corrects / (curr_count_sample * prediction_size), None)
                        else:
                            line = '\r' + construct_formatted_values_line(epoch, losses['train'][epoch], best_train_loss,
                                                                          current_loss, best_val_loss,
                                                                          time.time() - epoch_start_time, is_classification,
                                                                          last_acc['train'], running_corrects / (curr_count_sample * prediction_size))

                        sys.stdout.write(line)
                        sys.stdout.flush()

                    batch_iteration[phase] += 1

                if dataset_sizes[phase] > 0:
                    last_acc[phase] = running_corrects / (dataset_sizes[phase] * prediction_size)
                if dataset_sizes[phase] > 0:
                    losses[phase][epoch] = ObjectiveFunction.loss_from_running_loss(running_loss, dataset_sizes[phase])
                if tensorboard_logging:
                    tensorboard_writer.add_scalar('data/{}_loss'.format(phase), losses[phase][epoch], epoch)

                if tensorboard_logging:
                    for name, param in model.named_parameters():
                        tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                if (use_validation and phase == 'val' and losses['val'][epoch] < best_val_loss)\
                        or (not use_validation and phase == 'train' and losses['train'][epoch] < best_train_loss):
                    best_model_state_dict = model.state_dict()
                    best_val_loss = losses['val'][epoch]
                    best_train_loss = losses['train'][epoch]

            epoch_duration = time.time() - epoch_start_time
            try:
                criterion.next_epoch()
            except AttributeError:
                pass

            if verbose:
                line = '\r' + construct_formatted_values_line(epoch, losses['train'][epoch],
                                                              best_train_loss, losses['val'][epoch], best_val_loss,
                                                              epoch_duration, is_classification, last_acc['train'],
                                                              last_acc['val'], mem_usage=mem_usage)
                print_(line)

            if (use_validation and stopping_strategy.should_stop(losses['val'][epoch], epoch))\
                    or (not use_validation and stopping_strategy.should_stop(losses['train'][epoch], epoch)):
                losses['val'] = losses['val'][:epoch+1]
                losses['train'] = losses['train'][:epoch+1]
                break

        if visualize:
            plot_losses(losses)

        if verbose:
            print_()

    time_elapsed = time.time() - since

    model.load_state_dict(best_model_state_dict)

    snr = -1
    if verbose:
        print_()
        if dataset_sizes['val'] > 0:
            snrs = []
            if dataset.data_content in [DataContent.COMPLEX_PACKET_RD, DataContent.COMPLEX_RAMP]:
                snrs.append(evaluate_rd(model, dataloaders['val'].dataset, 'val'))
            if dataset.data_content in [DataContent.REAL_PACKET_RD]:
                snrs.append(evaluate_rd_log_mag(model, dataloaders['val'].dataset, 'val'))
            snr = np.mean(snrs)
        test_dataset = dataset.clone_for_new_active_partition(DatasetPartition.TEST)
        if len(test_dataset) > 0:
            if dataset.data_content in [DataContent.COMPLEX_PACKET_RD, DataContent.COMPLEX_RAMP]:
                evaluate_rd(model, test_dataset, 'test')
            if dataset.data_content in [DataContent.REAL_PACKET_RD]:
                evaluate_rd_log_mag(model, test_dataset, 'test')
        print_()

    evaluation_result = EvaluationResult(
        hyperparameters,
        task_id,
        task,
        best_train_loss,
        best_val_loss,
        time_elapsed,
        snr,
        JOB_DIR + '/model'
    )

    if tensorboard_logging:
        tensorboard_writer.close()

    return model, hyperparameters, evaluation_result


def construct_formatted_values_headline(is_classification=False):
    line = '\nepoch\t\ttloss\t\t\t\tvloss'
    if is_classification:
        line += '\t\t\ttacc\t\t\tvacc'
    if memory_logging:
        line += '\t\t\tmem'
    line += '\t\t\t\t\tduration'
    return line


def construct_formatted_values_line(epoch, train_loss, best_train_loss, val_loss, best_val_loss, epoch_duration, is_classification=False, train_acc=None, val_acc=None, mem_usage=None):
    line = '{}'.format(epoch + 1)
    value_text = '\t\t\t{:3.6f}'.format(train_loss)
    if train_loss <= best_train_loss and not is_cluster_run:
        line = line + PrintColors.GREEN + value_text + PrintColors.ENDC
    else:
        line = line + value_text

    if val_loss is not None:
        value_text = '\t\t\t{:3.6f}'.format(val_loss)
    else:
        value_text = '\t\t\t' + ' ' * 10
    if val_loss is not None and val_loss <= best_val_loss and not is_cluster_run:
        line = line + PrintColors.GREEN + value_text + PrintColors.ENDC
    else:
        line = line + value_text
    if is_classification:
        if train_acc is not None:
            line += '\t\t{:3.6f}'.format(train_acc)
        else:
            line += '\t\t' + ' ' * 10
        if val_acc is not None:
            line += '\t\t{:3.6f}'.format(val_acc)
        else:
            line += '\t\t' + ' ' * 10

    if mem_usage is not None:
        line += '\t\t{:6.2f} MB'.format(mem_usage)
    else:
        line += '\t\t' + ' ' * 10

    line += '\t\t{:10.2f}'.format(epoch_duration)

    return line
