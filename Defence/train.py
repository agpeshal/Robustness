import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils import Net, Normalize
from attacks import pgd_untargeted_batched


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default='42', help='seed')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='device')
parser.add_argument('--defense', type=bool, default=False, help='defense')
parser.add_argument('--num_epochs', type=int, default=1, help='epochs')
args = parser.parse_args()

# Setting the random number generator
torch.manual_seed(args.seed)

# Datasets
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), Net())
model = model.to(args.device)


opt = optim.Adam(params=model.parameters(), lr=args.learning_rate)
ce_loss = torch.nn.CrossEntropyLoss()


for epoch in range(1,args.num_epochs+1):
    t1 = time.time()

    # Training
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
        # Defence
        if args.defense:
            model.eval() # switch to evalaution mode
            print(y_batch)
            x_batch = pgd_untargeted_batched(model, x_batch, y_batch, k=5, eps=0.15,
                                         eps_step=0.05, device=args.device, clip_min=0, clip_max=1)
        model.train() # switch to training mode
        x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
        opt.zero_grad()
        out = model(x_batch)
        batch_loss = ce_loss(out, y_batch)
        batch_loss.backward()
        opt.step()
        
    # Testing
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
        x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()
        tot_acc += acc
        tot_test += x_batch.size()[0]
    t2 = time.time()

    print('Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (epoch, tot_acc/tot_test, t2-t1)) 


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/Net_{args.num_epochs}_{args.defense}")
