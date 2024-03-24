# Adversarial Machine Learning



This repository contains implementation of the Projected Gradient Descent (Madry *et al*) which is based on Fast Gradient Sign Method (Goodfellow *et al*) to attack neural networks trained on **CIFAR10** 

Defend the network by performed adversarial training

## Installation
Use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a virtual environment and install dependencies
```
conda create -n robust
conda activate robust
conda install --file requirements.txt
```


### Attack
Download model [checkpoint](https://drive.google.com/file/d/1cMzKCL3Woa-oZzz6gQyP-pYYJ-L9tPPw/view?usp=sharing)
To run PGD attack on model using CIFAR-10 images, run 
```bash
cd src
python attack.py --ckpt {path_to_ckpt}
```
Some examples of the adversarial images from the attack

![](images/car_truck.png) ![](images/frog_deer.png) 

 


### Defense

Use adversarial training as the defense mechanism

```bash
cd src
python train.py
```
 

### References

Goodfellow et al "[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)" ICLR, 2015

Aleksander Madry et al. ”[Towards deep learning models resistant to adversarial attacks](https://arxiv.org/abs/1706.06083)”. ICLR (2018).

