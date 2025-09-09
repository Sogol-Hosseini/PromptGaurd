from rich import print
import torch
import pandas as pd

print("[bold green]✓ Python OK[/bold green]")

# 1) Torch / GPU check
print(f"PyTorch version: {torch.__version__}")
gpu_ok = torch.cuda.is_available()
print(f"CUDA available: {gpu_ok}")
if gpu_ok:
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 2) Test Hugging Face dataset access
PATH = "hf://datasets/qualifire/prompt-injections-benchmark/test.csv"
print(f"\nReading sample from: {PATH}")
df = pd.read_csv(PATH, nrows=5)
print(df.head())

# 3) Quick column peek so we know what to map later
print("\nColumns:", list(df.columns))
