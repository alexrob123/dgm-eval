#!/bin/bash

# reference_ds='data/CIFAR10/CIFAR10/train'
# generated_ds='data/CIFAR10/CIFAR10-LOGAN'

reference_ds='data/ImageNet256/Imagenet256/train'
generated_ds='data/ImageNet256/Imagenet256-DiT-XL-2'

model='inception'

nearest_k="1 2 3 4 5 sqrt"

for k in $nearest_k
do
	echo 'Running with k =' $k
	if [ "$k" = "sqrt" ]; then
		k_arg=""
	else
		k_arg="--nearest-k $k"
	fi

    python -m dgm_eval \
	--train $reference_ds \
	--gen $generated_ds \
	--nsample -1 \
	--model $model \
	--save \
	--save-imgs \
	--metrics prdc \
	$k_arg \
	--per-label \
	--device cuda \
	--batch_size 256 \
	--output-dir experiments \
	--xp "sweep-prdc-k"

done
