# ConditionalGAN
A PyTorch implementation of ConditionalGAN.

## Related Papers

Alec Radford, Luke Metz and Soumith Chintala:
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
https://arxiv.org/abs/1511.06434

Mehdi Mirza and Simon Osindero: Conditional Generative Adversarial Nets. 
https://arxiv.org/pdf/1411.1784.pdf

## Training

```shell
% docker build -t cgan .
% ./train.sh
```

## Checking Training Results

```shell
% ./run_tensorboard.sh
```