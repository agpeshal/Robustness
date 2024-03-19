import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from attackers import PGD
from net.cnn_classifier import CNNClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--steps', type=int, default=10)

    return parser.parse_args()


def get_dataloader():
    return DataLoader(datasets.CIFAR10(
        '../datasets/cifar10',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    ), shuffle=True, batch_size=1)


def main():

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = CNNClassifier().to(device)
    model.eval()

    attacker = PGD(model, args.steps, args.alpha, args.eps, device)
    correct_adv, correct_org, total = 0, 0, 0
    dataloader = get_dataloader()

    for org_img, label in iter(dataloader):

        org_pred = torch.argmax(model(org_img).data, dim=1)
        adv_img = attacker.attack(org_img, label)
        output = model(adv_img)
        adv_pred = torch.argmax(output.data, dim=1)

        correct_adv += adv_pred == label
        correct_org += org_pred == label
        total += 1
    
    print("Accuracy on original images: {:.2f} %".format(correct_org / total * 100.0))
    print("Accuracy on adversarial images: {:.2f} %".format(correct_adv / total * 100.0))


if __name__ == '__main__':
    main()