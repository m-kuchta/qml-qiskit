"""Generate a list of command-line jobs for SLURM array execution.

This script creates a Cartesian product of the specified hyperparameter lists
(feature maps, ansatzes, entanglements, and seeds) and writes individual
CLI commands into 'jobs_array.txt'.
"""

import itertools
from pathlib import Path

PROJECT_ROOT = Path.cwd()
HPC_DIR = PROJECT_ROOT / "hpc"
OUTPUT_FILE = HPC_DIR / "jobs_array.txt"

HPC_DIR.mkdir(parents=True, exist_ok=True)

fmaps = ["Angle", "DenseAngle"] #, "Z", "ZZ", "PhaseCRY", "Amplitude"]
ansatzes = ["RealAmplitudes"] #, "EfficientSU2", "RY_RZ_CRX", "RY_RZ_RXX", "Overlapped", "DoubleEnt", "U_CU"]
entanglements = ["linear"]
seeds = [67]

combinations = list(itertools.product(fmaps, ansatzes, entanglements, seeds))

with open(OUTPUT_FILE, "w") as f:
        for fmap, ansatz, ent, seed in combinations:
            cmd = f"python run_vqc.py --fmap {fmap} --ansatz {ansatz} --fmap_entanglement {ent} --ansatz_entanglement {ent} --seed {seed}"
            f.write(cmd + "\n")

print(f"{len(combinations)} jobs were generated and saved in jobs_array.txt")
