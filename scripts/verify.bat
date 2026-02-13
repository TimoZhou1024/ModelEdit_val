uv run python scripts/verify_checkpoint.py --dataset pathmnist --model vit-base
uv run python scripts/verify_checkpoint.py --dataset dermamnist --model vit-base
@REM uv run python scripts/verify_checkpoint.py --dataset tissuemnist --model vit-base
@REM uv run python scripts/verify_checkpoint.py --dataset bloodmnist --model vit-base
uv run python scripts/verify_checkpoint.py --dataset organamnist --model vit-base
uv run python scripts/verify_checkpoint.py --dataset retinamnist --model vit-base
@REM uv run python scripts/verify_checkpoint.py --dataset liver4 --model vit-base
uv run python scripts/verify_checkpoint.py --dataset liver2a --model vit-base
uv run python scripts/verify_checkpoint.py --dataset liver2s --model vit-base

uv run python scripts/verify_checkpoint.py --dataset pathmnist --model vit-tiny
uv run python scripts/verify_checkpoint.py --dataset dermamnist --model vit-tiny
@REM uv run python scripts/verify_checkpoint.py --dataset tissuemnist --model vit-tiny
@REM uv run python scripts/verify_checkpoint.py --dataset bloodmnist --model vit-tiny
uv run python scripts/verify_checkpoint.py --dataset organamnist --model vit-tiny
uv run python scripts/verify_checkpoint.py --dataset retinamnist --model vit-tiny
@REM uv run python scripts/verify_checkpoint.py --dataset liver4 --model vit-tiny
uv run python scripts/verify_checkpoint.py --dataset liver2a --model vit-tiny
uv run python scripts/verify_checkpoint.py --dataset liver2s --model vit-tiny
