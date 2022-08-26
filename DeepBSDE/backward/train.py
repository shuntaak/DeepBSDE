import time
from solver import FeedForwardModel
import logging
import torch.optim as optim
import numpy as np
import os
import torch
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.backends.cudnn.benchmark=True

def train(config,bsde):
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)

    # build and train
    net = FeedForwardModel(config,bsde)
    net.cuda()

    optimizer = optim.SGD(net.parameters(),config.lr_values[0])
    start_time = time.time()
    # to save iteration results
    training_history = []
    # for validation
    dw_valid, x_valid = bsde.sample(config.valid_size)

    # begin sgd iteration
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.eval()
            loss,init = net(x_valid.cuda(), dw_valid.cuda())

            elapsed_time = time.time() - start_time
            training_history.append([step, loss.item(), init.item(), elapsed_time])
            if config.verbose:
                logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    step, loss, init, elapsed_time))

        dw_train, x_train = bsde.sample(config.batch_size)
        optimizer.zero_grad()
        net.train()
        loss, _ = net(x_train.cuda(), dw_train.cuda())
        loss.backward()

        optimizer.step()
    
    # training_history = training_history.detach().cpu().numpy()
    # for i in range(len(training_history)):
    #     training_history[i][1] = training_history[i][1].detach().cpu()
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
    name = sys.argv[1]
    cfg = get_config(name)
    bsde = get_equation(name, cfg.dim, cfg.total_time, cfg.num_time_interval)
    train(cfg,bsde)