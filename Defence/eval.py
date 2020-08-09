import argparse
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils import Net, Normalize
from attacks import pgd_untargeted


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='42', help='seed')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='device')
parser.add_argument('--defense', type=bool, default=False, help='defense')
parser.add_argument('--num_epochs', type=int, default=1, help='epochs')
args = parser.parse_args()

# Setting the random number generator
torch.manual_seed(args.seed)

# Setting up the Model
model = nn.Sequential(Normalize(), Net())
# Loading a previously trained one
model.load_state_dict(torch.load(f'./models/Net_{args.num_epochs}_{args.defense}'))
model.to(args.device)
model.eval()

# disable batches
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)


# number of samples we evaluate on
num_samples = 1000

xlist = [] # collect x tensors to use the same values in the next loop
ylist = [] # collect y tensors to use the same values in the next loop

# not attacked
###################################
# Evaluate on non-attacked images #
###################################

num_correct = 0
for _, (x, y) in enumerate(tqdm(itertools.islice(train_loader, num_samples))):
    xlist.append(x)
    ylist.append(y)
    x = x.to(args.device)
    y = y.to(args.device)
    out = model(x)
    pred = out.argmax(dim=1)
    num_correct += pred.eq(y).sum().item()

print('Accuracy %i samples original %.5lf' % (num_samples, num_correct / num_samples))

# attacked
###############################
# Evaluate on attacked images #
###############################

num_correct = 0
for x, y in tqdm(zip(xlist, ylist)):
    xperturbed = pgd_untargeted(model, x, y, k=10, eps=0.15, eps_step=0.08,
                   device=args.device, clip_min=0.0, clip_max=1.0)
    out = model(xperturbed)
    pred = out.argmax(dim=1)
    y = y.to(args.device)
    num_correct += pred.eq(y).sum().item()

print('Accuracy %i samples perturbed %.5lf' % (num_samples, num_correct / num_samples))
