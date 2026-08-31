"""Generate a list of command-line jobs for SLURM array execution.

This script creates a Cartesian product of the specified hyperparameter lists
(feature maps, ansatzes, entanglements, and seeds) and writes individual
CLI commands into 'jobs_array.txt'.
"""

import itertools

fmaps = ["ZZ", "Angle"]
ansatzes = ["Real_amplitudes", "EfficientSU2"]
entanglements = ["linear", "full"]
seeds = [1, 2, 3]

combinations = list(itertools.product(fmaps, ansatzes, entanglements, seeds))

with open("jobs_array.txt", "w") as f:
    for fmap, ansatz, ent, seed in combinations:
        cmd = f"python run_vqc.py --fmap {fmap} --ansatz {ansatz} --fmap_entanglement {ent} --ansatz_entanglement {ent} --seed {seed}"
        f.write(cmd + "\n")

print(f"{len(combinations)} jobs were generated and saved in jobs_array.txt")
