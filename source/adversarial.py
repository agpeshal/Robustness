import torch
from torch.nn import CrossEntropyLoss

class Adversarial:
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model.to(device)
        # set model to eval mode
        self.loss = CrossEntropyLoss()
    
    def FGSM(self, org_img, label, alpha, targeted=False):
        image = org_img.clone()
        # calculate gradients wrt the image
        image = image.to(self.device)
        image.requires_grad = True
        label = label.to(self.device)
        loss = self.loss(self.model(image), torch.tensor([label], dtype=torch.long))
        loss.backward()

        # zero model gradients
        self.model.zero_grad()
        delta = alpha * image.grad.sign()
        if targeted:
            return image - delta
        else:
            return image + delta
    

    def PGD(self, org_img, label, steps, alpha=2/255, eps=0.3, targeted=False):
        img_max = org_img + eps
        img_min = org_img - eps

        image = org_img
        for _ in range(steps):

            adv_img = self.FGSM(image, label, alpha, targeted=targeted)
            adv_img = torch.clamp(adv_img, min=img_min, max=img_max).detach_()
        
        return adv_img
        
