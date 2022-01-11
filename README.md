# ConditionalGAN
A PyTorch implementation of ConditionalGAN.

## Related Papers

Mehdi Mirza and Simon Osindero: Conditional Generative Adversarial Nets. 
https://arxiv.org/pdf/1411.1784.pdf

## Training

1. Clone this repository and move to the directory.

```shell
% clone https://github.com/mps-research/ConditionalGAN.git
% cd ConditionalGAN
```

2. At the repository root directory, build "cgan" docker image and run the image inside of a container.

```shell
% docker build -t cgan .
% ./train.sh
```

## Checking Training Results

At the repository root directory, execute the following command.

```shell
% ./run_tensorboard.sh
```
