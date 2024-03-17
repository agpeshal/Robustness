import torch
from torch.nn import CrossEntropyLoss

class Adversarial:
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model.to(device)
        # set model to eval mode
        self.loss = CrossEntropyLoss()
    
    def FGSM(self, org_img, label, alpha, targeted=False):
        # calculate gradients wrt the image
        org_img = org_img.to(self.device)
        org_img.requires_grad = True
        label = label.to(self.device)
        loss = self.loss(self.model(org_img), torch.tensor([label], dtype=torch.long))
        loss.backward()

        # zero model gradients
        self.model.zero_grad()
        delta = alpha * self.image.grad.sign()
        if targeted:
            return self.image - delta
        else:
            return self.image + delta
    

    def PGD(self, org_img, label, steps, alpha=2/255, eps=0.3, targeted=False):
        img_max = org_img + eps
        img_min = org_img - eps

        image = org_img
        for _ in range(steps):

            adv_img = self.FGSM(image, label, alpha, targeted=targeted)
            adv_img = torch.clamp(adv_img, min=img_min, max=img_max).detach_()
        
        return adv_img
        
