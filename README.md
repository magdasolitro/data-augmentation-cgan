# Conditional GANs for Data Augmentation to improve side-channel attacks
(currently under editing)
Carrying out power analysis on a cryptographic device with deep learning techniques is one of the latest frontiers of side-channel attacks. 
However, deep learning algorithms are data-hungry, and this approach is sometimes difficult to apply due to the lack of data needed to train the network. 
One possible way to overcome this obstacle is Data Augmentation, which means increasing the amount of training samples, by adding newly created synthetic 
data from already existing samples. One way to do this is to use Generative Adversarial Networks, an architecture introduced in 2014 that proved to be 
exceptionally good at producing fake data in the context of image generation.
For this project, I used a variant of the traditional GAN architecture, called Conditional Generative Adversarial Network (cGANs), that allows to provide an
additional input (a class label, in this case) to orient the generation of the power trace.

## Architectures
The repository contains two different architectures (that go under the names of Model A and Model B), that were developed based on two recent papers on the 
topic.

### Model A 
The first model was taken from *"Fake it till you make it: Data Augmentation using Generative Adversarial Networks for all the crypto you need 
on small devices"* by Naila Mukhtar, Lejla Batina, Stjepan Picek, and Yinan Kong. 

### Model B
From *Data Augmentation with Conditional GAN for Automatic Modulation Classification*, by Mansi Patel, Xuyu Wang, and Shiwen Mao.
