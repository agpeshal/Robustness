from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Attacker(ABC):
    @abstractmethod
    def attack(self, org_img, label):
        ...


class FGSM(Attacker):
    def __init__(
        self,
        model: nn.Module,
        alpha: float,
        targeted: bool = False,
    ) -> None:
        """Fast Gradient Sign Method

        https://arxiv.org/abs/1412.6572

        Args:
            model (nn.Module): Model to be attacked
            alpha (float): max change in the image per pixel
            targeted (bool, optional): If True, resulting image will force model to predict the provided label. Defaults to False.
        """
        self.model = model
        self.alpha = alpha
        self.targeted = targeted

    def attack(self, org_img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Attack image with the given label

        Args:
            org_img (torch.Tensor): Target image
            label (torch.Tensor): True label if self.targeted is set to False else, target label.

        Returns:
            torch.Tensor: Adversarial image
        """
        # set model in eval mode
        self.model.eval()

        # clone original image before modifying
        image = org_img.clone()

        # calculate gradients wrt the image
        image.requires_grad = True
        loss = nn.CrossEntropyLoss()(
            self.model(image), torch.tensor([label], dtype=torch.long)
        )
        loss.backward()

        # zero model gradients
        self.model.zero_grad()

        # move in the direction of the sign of the gradient
        assert image.grad is not None
        delta = self.alpha * image.grad.sign()
        if self.targeted:
            return image - delta
        else:
            return image + delta


class PGD(Attacker):
    def __init__(
        self,
        model: nn.Module,
        steps: int,
        alpha: float = 2 / 255,
        eps: float = 0.3,
        targeted=False,
    ) -> None:
        """Projected Gradient Sign Method

        https://arxiv.org/abs/1706.06083

        Args:
            model (nn.Module): Model to attack
            steps (int): _description_
            alpha (float, optional): Max per pixel change in each step of attack. Defaults to 2/255.
            eps (float, optional): Max per pixel change in the final output. Defaults to 0.3.
            targeted (bool, optional):  If True, resulting image will force model to predict the provided label. Defaults to False.
        """
        self.fgsm = FGSM(model, alpha, targeted)
        self.steps = steps
        self.alpha = alpha
        self.eps = eps

    def attack(self, org_img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Attack image with the given label

        Args:
            org_img (torch.Tensor): Target image
            label (torch.Tensor): True label if self.targeted is set to False else, target label.

        Returns:
            torch.Tensor: Adversarial image
        """
        img_max = org_img + self.eps
        img_min = org_img - self.eps

        image = org_img
        for _ in range(self.steps):
            adv_img = self.fgsm.attack(image, label)
            adv_img = torch.clamp(adv_img, min=img_min, max=img_max).detach_()

        return adv_img
