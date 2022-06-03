# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model.basic import rao_gumbel, gst_mover

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                    help='softmax temperature (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
#parser.add_argument('--hard', type=lambda x: str(x).lower()=='true', default=True,
parser.add_argument('--hard', type=str, default='True',
                    help='hard Gumbel Softmax')
parser.add_argument('--mode', default='gumbel',
                    help='gumbel, st, rao_gumbel, gst-p, gst-1.0')
parser.add_argument('--log-images', type=lambda x: str(x).lower()=='true', default=False, 
                    help='log the sample & reconstructed images')
parser.add_argument('--log-test', type=lambda x: str(x).lower()=='true', default=False, 
                    help='log the testing error into files')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def categorical_repara(logits, temp, 
                       hard=True, mode = 'gumbel', training=True):
    if mode == 'gumbel':
        return F.gumbel_softmax(logits, tau=temp, hard=hard).view(-1, latent_dim * categorical_dim)
    
    elif mode == 'st':
        m = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        action = m.sample()
        prob = (logits/temp).softmax(dim=-1)
        action = action - prob.detach() + prob if hard else prob
        return action.view(-1, latent_dim * categorical_dim)    
    
    elif mode.startswith('gst-'):
        try:
            gap = float(mode[4:])
            if gap<0.0: gap = 'p'
        except:
            gap = 'p'
        ret = gst_mover(logits, temp, hard=hard, gap=gap)
        return ret.flatten(1)
    
    elif mode.startswith('nd_gst-'):
        try:
            gap = float(mode[7:])
            if gap<0.0: gap = 'p'
        except:
            gap = 'p'
        ret = gst_mover(logits, temp, hard=hard, gap=gap, detach=False)
        return ret.flatten(1)

    elif mode == 'rao_gumbel': # rao-blackwellizing
        ret = rao_gumbel(logits, temp, None, repeats=100, hard=hard)
        return ret.flatten(1)
    else:
        print(mode + ' not supported')
        exit()

class VAE_gumbel(nn.Module):
    def __init__(self):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hard = True
        self.mode = 'gumbel'

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, with_z=False):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        z = categorical_repara(q_y, temp, self.hard, self.mode, self.training)
        qy = F.softmax(q_y, dim=-1).reshape(*q.size())
        if not with_z:
            return self.decode(z), qy
        else:
            return self.decode(z), qy, z


latent_dim = 30
categorical_dim = 10  # one-of-K vector

model = VAE_gumbel()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.hard = args.hard.lower() == 'true'
model.mode = args.mode

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp)
        loss = loss_function(recon_batch, data, qy)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, log_images=False, out=None):
    model.eval()
    test_loss = 0
    temp = args.temperature
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp)
        test_loss += loss_function(recon_batch, data, qy).item() * len(data)
        if i == 0 and log_images:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'data/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if out is not None:
        out.write('{:.4f}\n'.format(test_loss))
        out.flush()

def run():
    if args.log_test:
        if not os.path.isdir('vae_results'):
            os.mkdir('vae_results')
        out = open('vae_results/{}_{}_{}_{}.txt'.format(args.mode, args.temperature, args.seed, args.hard), 'w')
    else: out = None

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch, args.log_images, out)
        
        if args.log_images:
            M = 64 * latent_dim
            np_y = np.zeros((M, categorical_dim), dtype=np.float32)
            np_y[range(M), np.random.choice(categorical_dim, M)] = 1
            np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
            sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
            if args.cuda: sample = sample.cuda()
            sample = model.decode(sample).cpu()
            save_image(sample.data.view(M // latent_dim, 1, 28, 28),
                       'data/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    run()
