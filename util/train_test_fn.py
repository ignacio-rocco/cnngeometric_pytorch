from __future__ import print_function, division
from tqdm import tqdm


def train(epoch, model, loss_fn, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50, scheduler=False):

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        train_loss += loss.data.cpu().numpy().item()
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test(model, loss_fn,
         dataloader, pair_generation_tnf):

    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy()[0]

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss
