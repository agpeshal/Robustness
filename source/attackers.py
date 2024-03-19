from abc import ABC, abstractmethod
import torch
from torch.nn import CrossEntropyLoss

class Attacker(ABC):

    @abstractmethod
    def attack(self, org_img, label):
        ...

class FGSM(Attacker):
    def __init__(self, model, alpha, device='cpu', targeted=False) -> None:
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.targeted = targeted

    def attack(self, org_img, label):
        # set model in eval mode
        self.model.eval()

        # clone original image before modifying
        image = org_img.clone()
        
        # calculate gradients wrt the image
        image = image.to(self.device)
        image.requires_grad = True
        label = label.to(self.device)
        loss = CrossEntropyLoss()(self.model(image), torch.tensor([label], dtype=torch.long))
        loss.backward()

        # zero model gradients
        self.model.zero_grad()
        
        # move in the direction of the sign of the gradient
        delta = self.alpha * image.grad.sign()
        if self.targeted:
            return image - delta
        else:
            return image + delta


class PGD(Attacker):
    def __init__(self, model, steps, alpha=2/255, eps=0.3, device='cpu', targeted=False) -> None:
        self.fgsm = FGSM(model, alpha, device, targeted)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.eps = eps
    
    def attack(self, org_img, label):
        img_max = org_img + self.eps
        img_min = org_img - self.eps

        image = org_img
        for _ in range(self.steps):

            adv_img = self.fgsm.attack(image, label)
            adv_img = torch.clamp(adv_img, min=img_min, max=img_max).detach_()
        
        return adv_img


    