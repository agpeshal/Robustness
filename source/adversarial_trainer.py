import torch
from tqdm import tqdm


class AdversarialTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        attacker,
        loss,
        epochs,
        eval_interval,
        optimizer,
        device="cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attacker = attacker
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def train(self):
        for i in range(self.epochs):
            self.train_epoch_()

            if i % self.eval_interval == 0:
                self.eval()

    def eval(self):
        self.model.eval()
        correct, total = 0, 0
        for images, labels in tqdm(iter(self.test_loader), desc="Evaluating ..."):
            images = images.to(self.device)
            labels = labels.to(self.device)
            predictions = self.predict_(images).detach()
            correct += (labels == predictions).sum().cpu().numpy()
            total += len(labels)

        print("Test accuracy: {:.2f}".format(correct / total * 100.0))

    def predict_(self, images):
        return torch.argmax(self.model(images).data, dim=1)

    def train_epoch_(self):
        self.model.train()
        for images, labels in iter(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            adv_images = self.attacker.attack(images, labels)
            self.optimizer.zero_grad()
            outputs = self.model(adv_images)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
