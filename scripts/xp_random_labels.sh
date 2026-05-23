#!/bin/bash

# reference_ds='data/CIFAR10/CIFAR10/train'
# generated_ds='data/CIFAR10/CIFAR10-LOGAN'

reference_ds='data/ImageNet256/Imagenet256/train'
generated_ds='data/ImageNet256/Imagenet256-DiT-XL-2'

model='inception'

echo 'Running experiment: Random Labels'

echo 'Running with randomized labels...'
python -m dgm_eval \
	--train $reference_ds \
	--gen $generated_ds \
	--nsample -1 \
	--model $model \
	--save \
	--metrics prdc \
	--per-label \
	--device cuda \
	--batch_size 256 \
	--output-dir experiments \
	--xp "random-labels"

echo 'Running with original labels...'
python -m dgm_eval \
	--train $reference_ds \
	--gen $generated_ds \
	--nsample -1 \
	--model $model \
	--save \
	--metrics prdc \
	--per-label \
	--device cuda \
	--batch_size 256 \
	--output-dir experiments