# run_5060 Runtime Snapshot

Captured on 2026-07-29 (Asia/Shanghai) after the validation-protocol smoke passed.

- Conda prefix: `D:\miniconda\envs\run_5060`
- OS: Microsoft Windows 10.0.26200, x64
- Python: 3.10.20
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- GPU: NVIDIA GeForce RTX 5060, 8151 MiB
- NVIDIA driver: 595.97
- DGL package: 2.2.1
- NumPy: 2.2.6
- SciPy: 1.15.3
- scikit-learn: 1.7.2

The active `codes/main_mmlight.py` entry path imports and trains successfully through the repository's Windows compatibility handling. A raw standalone `import dgl` still encounters the known missing GraphBolt DLL for PyTorch 2.11; reproduce the project through its active entry path unless that packaging mismatch is fixed separately.

Restore order:

1. Create the base Conda environment from `run_5060-conda-explicit.txt`.
2. Install the packages pinned in `run_5060-pip-freeze.txt` using the CUDA/PyTorch indexes appropriate for the target machine.
3. Verify the GPU driver and run the repository smoke before starting a long experiment.

