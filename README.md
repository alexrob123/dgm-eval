<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Evaluation of Deep Generative models

The codebase for evaluation of deep generative models as presented in [_Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models_](https://arxiv.org/abs/2306.04675), accepted to [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73076)

We studied 41 generative models across a diverse range of image datasets and found:

- The state-of-the-art perceptual realism of diffusion models as judged by humans is not reflected in commonly reported metrics when using the default Inception-V3 network.
- Supervised networks do not provide a perceptual space that generalizes well for image evaluation, and neither do self-supervised methods from particular families.
- [DINOv2](https://github.com/facebookresearch/dinov2) provides such a generalized representation space and allows for much richer evaluation of generative models. **Researchers should replace Inception-V3 in all evaluation metrics**. We provide an extensive DINOv2 leaderboard [below](#dinov2-leaderboard) and have added the results to _paperswithcode.com_.
- Generative models directly memorize training examples on simple, smaller datasets like CIFAR10, but not necessarily on more complex datasets like ImageNet. However, our experiments show that currently proposed diagnostic metrics do not properly detect memorization.

Here we provide code to compute the following 15 generative evaluation metrics using 8 different encoder networks:

Metrics:

- Fréchet Distance: [FD](https://arxiv.org/abs/1706.08500)
- [FD<sub>∞</sub>](https://arxiv.org/abs/1911.07023)
- Spatial FID: [sFID](https://arxiv.org/abs/2103.03841)
- [Kernel Distance](https://arxiv.org/abs/1801.01401)
- [Inception Score](https://arxiv.org/abs/1606.03498)
- [FLS](https://arxiv.org/abs/2302.04440)
- [Precision & Recall](https://arxiv.org/abs/1904.06991)
- [Density & Coverage](https://arxiv.org/abs/2002.09797)
- [Vendi score](https://arxiv.org/abs/2210.02410)
- [AuthPct](https://arxiv.org/abs/2102.08921)
- [C<sub>T</sub> score](https://arxiv.org/abs/2004.05675)
- [FLS-POG](https://arxiv.org/abs/2302.04440)
- [Realism](https://arxiv.org/abs/1904.06991)
- Approximate Sliced Wasserstein: [ASW](https://arxiv.org/abs/2106.15427)

Encoders:

- Inception
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [SimCLRv2](https://github.com/google-research/simclr)
- [SwAV](https://github.com/facebookresearch/swav/)
- [CLIP](https://github.com/mlfoundations/open_clip)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MAE](https://github.com/facebookresearch/mae)
- [data2vec](https://ai.facebook.com/blog/ai-self-supervised-learning-data2vec/)


## Table of contents
[Installation & Usage](#installation--usage)  
[Data Access](#data-access)  
[Visualizing Heatmaps](#visualizing-heatmaps)  
[References](#references)  
[License](#license)

## Installation & Usage

### Installation

First clone this repository, then navigate to the directory and pip install to install all required packages.

```
git clone git@github.com:layer6ai-labs/dgm-eval
cd dgm-eval
pip install -e .
```

We recommend you do this in a conda environment:

```
conda create --name dgm-eval pip python==3.10
conda activate dgm-eval
git clone git@github.com:layer6ai-labs/dgm-eval
cd dgm-eval
pip install -e .
```

### Usage

Computing metrics only requires the paths to either locally hosted image datasets or torchvision.datasets. Encoders are automatically downloaded. For example, the following will compute the Fréchet distance (fd), kernel distance (kd), precision/recall/density/coverage (prdc), and the C<sub>T</sub> score (ct) using DINOv2 (default) as the encoder.

```
python -m dgm_eval \
	--train path/to/training_dataset \
	--gen path/to/generated_dataset \
	--test_path path/to/test_dataset \
	--model dinov2 \
	--metrics fd kd prdc ct
```

See `scripts/run_experiments.sh` or run `python dgm_eval -h` for further details on commandline parameters. As we suggest in the paper, metrics should be reported using the default model size (DINOv2-ViT-L/14) for final leaderboard values, but tracking progress during training is a factor of 4 more efficient with DINOv2-ViT-B/14. To use this architecture instead simply add `--arch vitb14` as a commandline parameter.

Local datasets should either be un-conditional:

```
local/path/
	000000.png
	000001.png
	...
```

or conditional:

```
local/path/
	0/
		000000.png
		000001.png
		...
	1/
		000000.png
		000001.png
		...
	...
```

The directory should only include image files. To download and use a dataset from torchvision.datasets, just specify the dataset and train/test string:

```
python -m dgm_eval \
	--train CIFAR10--train \
	--gen CIFAR10--test
```

A full example is as follows:

```
python -m dgm_eval \
	--train CIFAR10--train \
	--gen CIFAR10--test \
	--model dinov2 \
	--metrics fd kd prdc \
	--device cuda \
	--batch_size 256 \
	--nsample 512


>>> ....
>>> Num real: 512 Num fake: 512
>>> fd: 862.53745
>>> kd_value: 0.01095
>>> kd_variance: 0.00000
>>> precision: 0.90430
>>> recall: 0.91797
>>> density: 0.97969
>>> coverage: 0.94141
```

## Data Access

### Images

All generated data shown in this work can be accessed at the following link:

[drive.google.com/drive/folders/1X0MFaUta90d3zF9xG4KchjR-8SE0cT_7?usp=sharing](https://drive.google.com/drive/folders/1X0MFaUta90d3zF9xG4KchjR-8SE0cT_7?usp=sharing)

Including:

- Datasets of 100,000 image samples from 41 generative models across `CIFAR10/`, `imagenet256/`, `LSUN Bedroom/`, and `FFHQ256/`.
- Training & test data at 256 x 256 resolution
- Generated datasets for controlled experiments presented in the Appendix can be found in `toy-datasets/`


**CIFAR10**

| Dataset | Imgs | Classes | Img per class |
|---|---|---|---|
| CIFAR10/train | 50k | 10 | 5k |
| CIFAR10/test | 10k | 10 | 1k |
| CIFAR10_LOGAN | 100k | 10 | 10k |

**ImageNet256**

| Dataset | Imgs | Classes | Img per class |
|---|---|---|---|
| ImageNet256/train | 100k | 1k | 100 |
| ImageNet256/val | 50k | 1k | 50 |
| Imagenet256_DiT_XL_2 | 100k | 1k | 100 |



### Human Evaluation

Data for human evaluation of image realism can be found at `data/human-evaluation-realism/`


## Visualizing Heatmaps

Heatmaps can be visualized for each model on any given image datasets by the following, with examples following:

```
python -m dgm_eval CIFAR10--train CIFAR10--test \
					 --model inception \
					 --metrics fd \
					 --device cuda \
					 --batch_size 256 \
					 --nsample 50000 \
					 --heatmaps
```

|                   Images                   |                 Inception                  |                 DINOv2                  |
| :----------------------------------------: | :----------------------------------------: | :-------------------------------------: |
| ![image](figures/heatmaps_inception_1.png) | ![image](figures/heatmaps_inception_2.png) | ![image](figures/heatmaps_dinov2_2.png) |

## References

```
@inproceedings{stein2023exposing,
  title={Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models},
  author={Stein, George and Cresswell, Jesse and Hosseinzadeh, Rasa and Sui, Yi and Ross, Brendan and Villecroze, Valentin and Liu, Zhaoyan and Caterini, Anthony L and Taylor, Eric and Loaiza-Ganem, Gabriel},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
