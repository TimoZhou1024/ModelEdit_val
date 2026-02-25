#!/bin/bash

# uv run python src/main_lwe.py --dataset pathmnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_pathmnist

# uv run python src/main_lwe.py --dataset dermamnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_dermamnist

# uv run python src/main_lwe.py --dataset bloodmnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_bloodmnist

# uv run python src/main_lwe.py --dataset organamnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_organamnist

# uv run python src/main_lwe.py --dataset retinamnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_retinamnist


uv run python src/main_lwe.py --dataset organamnist --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_organamnist_2

# uv run python src/main_lwe.py --dataset liver2s --model vit-tiny --lwe-method hprd --lwe-hprd-epochs 10 --max-edits 3000 --run-name lwe_exp_liver2s