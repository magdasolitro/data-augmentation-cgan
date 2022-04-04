# Conditional GANs for Data Augmentation to improve side-channel attacks
(currently under editing)
Carrying out power analysis on a cryptographic device with deep learning techniques is one of the latest frontiers of side-channel attacks. 
However, deep learning algorithms are data-hungry, and this approach is sometimes difficult to apply due to the lack of data needed to train the network. 
One possible way to overcome this obstacle is Data Augmentation, which means increasing the amount of training samples, by adding newly created synthetic 
data from already existing samples. 
In this repository, you can find two Conditional Generative Adversarial Networks (cGANs) that I implemented with the goal of generating new power traces. 

## Architectures
The repository contains two different architectures (that go under the names Model A and Model B), found in two recent papers. 

### Model A 
The first model was taken from *"Fake it till you make it: Data Augmentation using Generative Adversarial Networks for all the crypto you need 
on small devices"* by Naila Mukhtar, Lejla Batina, Stjepan Picek, and Yinan Kong. The authors gave only a high-level description of the architecture,  

### Model B
From *Data Augmentation with Conditional GAN for Automatic Modulation Classification*, by Mansi Patel, Xuyu Wang, and Shiwen Mao.
