import time
from solver import FeedForwardModel
import logging
import torch.optim as optim
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

def __optimize_cuda_visible_devices() -> None:
    """Sets CUDA_VISIBLE_DEVICES so as to enable PyTorch to choose GPUs in the
    descending order of free GPU-memory size.  This must be called before
    importing PyTorch.
    """

    import csv
    import os
    import subprocess

    command = [
        "nvidia-smi",
        "--query-gpu=memory.free,index",
        "--format=csv,noheader,nounits",
    ]
    csv_rows = subprocess.check_output(command).decode("utf8").splitlines()
    rows = list(tuple(int(c.strip()) for c in r) for r in csv.reader(csv_rows))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(r[1]) for r in sorted(reversed(rows), key=lambda r: -int(r[0]))
    )


# Set CUDA_VISIBLE_DEVICES if possible.  Ignone any exceptions because this is
# not mandatory.
try:
    __optimize_cuda_visible_devices()
except Exception:
    pass

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.backends.cudnn.benchmark=True


# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.normal_(std=5.0/np.sqrt(cin+cout))

def train(config,bsde):
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)

    # build and train
    net = FeedForwardModel(config,bsde)
    net.cuda()
    optimizer = optim.SGD(net.parameters(),config.lr_values[0]
                         )    # optimizer = optim.SGD(net.parameters(),5e-4)
    start_time = time.time()
    # to save iteration results
    training_history = []
    # for validation
    dw_valid, x_valid = bsde.sample(config.valid_size)
    shape=[]

    # begin sgd iteration
    for t in range(config.num_time_interval): #for each timestep
        # net.apply(init_weights)
        # net = FeedForwardModel(config,bsde,target,config.num_time_interval-t)
        # net.cuda()
        # optimizer = optim.SGD(net.parameters(),5e-4)
        if config.verbose:
            logging.info("training of the subnetwork %5u start" % (config.num_time_interval-t))
        num_iter = config.num_iterations + 1
        if t == 0:
            num_iter=10000
        for step in range(num_iter):
            if step % config.logging_frequency == 0:
                # net.eval()
                loss,init = net.eval_loss(x_valid.cuda(), dw_valid.cuda())
                # init = net.output(x_valid.cuda())
                # print(loss, init)
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, init, elapsed_time])
                if config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, init, elapsed_time))
            
            dw_train, x_train = bsde.sample(config.batch_size)
            optimizer.zero_grad()
            net.train()
            loss = net.forward(x_train.cuda(), dw_train.cuda())
            loss.backward()
            # if net._t == 0:
                # print(list(net._subnetwork_value.parameters()))
            optimizer.step()
        shape.insert(0,init)
        # target = copy.deepcopy(net._subnetwork(requires_grad = False))
        net.target_update()
        net.cuda()
        optimizer = optim.SGD(net.parameters(),config.lr_values[0])
        if config.verbose:
            logging.info("training of the subnetwork %5u done" % (config.num_time_interval-t))
    plt.plot(np.arange(config.num_time_interval)*config.total_time/config.num_time_interval, np.array(shape))
    plt.savefig("output_shape{}.pdf".format(name))
    
    # training_history = training_history.detach().cpu().numpy()
    for i in range(len(training_history)):
        training_history[i][1] = training_history[i][1].detach().cpu()
    training_history =np.array(training_history)

    if bsde.y_init:
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(
                         abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init))


    np.savetxt('{}_training_history.csv'.format(bsde.__class__.__name__),
                training_history,
                fmt=['%d', '%.5e', '%.5e', '%d'],
                delimiter=",",
                header="step,loss_function,target_value,elapsed_time",
                comments='')
    

if __name__ == '__main__':
    from config import get_config
    from equation import get_equation
    torch.no_grad()
    name = sys.argv[1]
    cfg = get_config(name)
    bsde = get_equation(name, cfg.dim, cfg.total_time, cfg.num_time_interval)
    train(cfg,bsde)