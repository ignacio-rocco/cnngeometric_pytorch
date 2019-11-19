from __future__ import print_function, division
import numpy as np
from tqdm import tqdm


def train(epoch, model, loss_fn, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50, tb_writer=None, scheduler=False):
    """
    Main function for training

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param optimizer: optimizer of the model
    :param dataloader: DataLoader object
    :param pair_generation_tnf: Function to serve couples of samples
    :param log_interval: int, number of steps before logging scalars
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param scheduler: Eventual Learning rate scheduler

    :return: float, avg value of loss fn over epoch
    """

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            loss = loss_fn(theta, np.reshape(tnf_batch['theta_GT'], [16, 6]))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        train_loss += loss.data.cpu().numpy().item()

        # log every log_interval
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))
            if tb_writer:
                tb_writer.add_scalar('training loss',
                                     loss.data.item(),
                                     epoch * len(dataloader) + batch_idx)

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test(model, loss_fn,
         dataloader, pair_generation_tnf,
         epoch, tb_writer=None):

    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy()[0]

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    if tb_writer:
        tb_writer.add_scalar('training loss',
                             test_loss.data.item(),
                             epoch)

    return test_loss
