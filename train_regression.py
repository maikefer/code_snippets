import os
import torch

from torch.utils.tensorboard import SummaryWriter

from regression_net.metrics import bce_only_tm, bce_only_tv, weighted_mse_rt_cos, weighted_mse_rt_sin, \
    weighted_mse_ab, accuracy_tm, accuracy_tv, mse_only_coordinates
from regression_net.regression_loss import MyRegressionLoss
from regression_net.models.loader import load_regression_model
from utilities import training, io_utilities
from utilities.profiler import Profiler


def train(config):
    # use my own profiler b.c. profiling rights are not enabled on server cluster
    my_own_profiler = Profiler()
    my_own_profiler.begin_profiling()

    # setup experiment and log folder
    folder, prefix_id = io_utilities.create_experiment_folder(config)
    print('>>>> Folder of the Experiment: ', folder)
    io_utilities.dump_config(config, folder + prefix_id)

    run_folder = folder + 'run/'
    os.mkdir(run_folder)
    writer = SummaryWriter(log_dir=run_folder)

    # setup data
    my_own_profiler.start('data_loading')
    test_data_loader, train_data_loader = load_data(config, label_name='coordinates')
    my_own_profiler.end('data_loading')

    # setup model
    my_own_profiler.start('model_loading')
    model, previous_epochs = load_regression_model(config)
    my_own_profiler.end('model_loading')

    if torch.cuda.device_count() > 1:
        print("You are training on:", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # setup training
    my_own_profiler.start('training')
    model_file_name = folder + prefix_id + config['model']['name'] + '_model'

    metrics = ([bce_only_tm, accuracy_tm,
                bce_only_tv, accuracy_tv,
                mse_only_coordinates,
                weighted_mse_ab, weighted_mse_rt_sin, weighted_mse_rt_cos],
               ['BCE-TM', 'ACCURACY-TM',
                'BCE-TV', 'ACCURACY_TV',
                'MSE-COORDS',
                'Weighted-MSE-AB', 'Weighted-MSE-RT-Sin', 'Weighted-MSE-RT-Cos'])

    loss_criterion = MyRegressionLoss(get_bce_weights(config)).calc_loss

    optimizer = torch.optim.Adam(model.parameters())
    number_down_epochs = 2
    number_up_epochs = 2
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['lr'][0],
                                                  max_lr=config['lr'][1],
                                                  step_size_up=number_up_epochs * len(train_data_loader),
                                                  step_size_down=number_down_epochs * len(train_data_loader),
                                                  cycle_momentum=False)
    # start training
    training.fit(model=model,
                 criterion=(loss_criterion, 'MSE-BCE'),
                 metrics=metrics,
                 optimizer=optimizer,
                 lr_scheduler=scheduler,
                 train_data_loader=train_data_loader,
                 test_data_loader=test_data_loader,
                 my_own_profiler=my_own_profiler,
                 model_file_name=model_file_name,
                 previous_epochs=previous_epochs,
                 training_folder=folder,
                 config=config,
                 summary_writer=writer)

    my_own_profiler.end('training')
    my_own_profiler.stop_profiling()
    my_own_profiler.print()


def get_bce_weights(config):
    """
    Retrieve BCE weights from config or use default parameter for older trainings
    :param config: config dict, should contain 'bce_weight'
    :return: BCE weights for loss function
    """
    return [1, 1] if 'bce_weights' not in config.keys() else config['bce_weights']


if __name__ == '__main__':
    args = io_utilities.get_console_arguments()
    config = io_utilities.get_config(args)
    train(config)
