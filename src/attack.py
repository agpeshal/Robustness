import argparse

import torch

from core.attackers import PGD
from data.dataloaders import get_dataloader
from models.cnn_classifier import CNNClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNClassifier().to(device)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint)

    model.eval()

    attacker = PGD(model, args.steps, args.alpha, args.eps, device)
    correct_adv, correct_org, total = 0, 0, 0
    dataloader = get_dataloader(batch_size=1, train=False, shuffle=False)

    for org_img, label in iter(dataloader):
        org_img = org_img.to(device)
        label = label.to(device)
        org_pred = torch.argmax(model(org_img).data, dim=1)
        adv_img = attacker.attack(org_img, label)
        output = model(adv_img)
        adv_pred = torch.argmax(output.data, dim=1)

        correct_adv += adv_pred == label
        correct_org += org_pred == label
        total += 1

    clean_acc = correct_org / total * 100.0
    adv_acc = correct_adv / total * 100.0
    print(f"Accuracy on original images: {clean_acc:.2f} %")
    print(f"Accuracy on adversarial images: {adv_acc:.2f} %")


if __name__ == "__main__":
    main()
