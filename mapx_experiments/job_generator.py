import os
from pathlib import Path

# ====================================================
# 1. USER CONFIGURATION
# ====================================================

# ----------------- PATHS -----------------
CODE_DIR = Path("/nfs/stak/users/viswansi/hpc-share/morl-baselines-env/morl-baselines/mapx_experiments")
EXPERIMENT_TAG = "morld_benchmark_v1"
JOB_SCRIPTS_DIR = Path(f"/nfs/stak/users/viswansi/hpc-share/morl-baselines-env/morl-baselines/mapx_experiments/jobs/{EXPERIMENT_TAG}")
CONDA_ENV_PATH = "/nfs/stak/users/viswansi/hpc-share/morl-baselines-env"
# The folder containing the actual 'morl_baselines' source code package
PROJECT_ROOT = Path("/nfs/stak/users/viswansi/hpc-share/morl-baselines-env/morl-baselines")

# ----------------- WANDB SETUP -----------------
WANDB_API_KEY = ""
WANDB_ENTITY = ""
WANDB_PROJECT = "MORL-Baselines"

# ----------------- EXPERIMENT CONFIGS -----------------
# List of seeds to run for EACH environment
SEEDS = [2024, 2025]

ENV_SETTINGS = {
    # "mo-ant-2obj-v5": {
    #     "timesteps": 8000000,
    #     "ref_point": "-10000.0 -10000.0" 
    # },
    # "mo-walker2d-v5": {
    #     "timesteps": 8000000,
    #     "ref_point": "-10000.0 -10000.0" 
    # },
    "mo-swimmer-v5": {
        "timesteps": 2000000,
        "ref_point": "-10000.0 -10000.0" 
    },
    "mo-hopper-2obj-v5": {
        "timesteps": 8000000,
        "ref_point": "-10000.0 -10000.0" 
    },
    "mo-ant-2obj-v5": {
        "timesteps": 8000000,
        "ref_point": "-10000.0 -10000.0" 
    },
}

# ----------------- SLURM TEMPLATE -----------------
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/error.log
#SBATCH --output={log_dir}/output.out
#SBATCH --time=0-48:00:00
#SBATCH --partition=dgx2,dgxh,share,ampere
#SBATCH --constraint=skylake
#SBATCH --mem=32G
#SBATCH -c 12

# 1. WandB Setup
export WANDB_API_KEY={wandb_key}
# export WANDB_ENTITY={wandb_entity}
export WANDB_PROJECT={wandb_project}
export WANDB_DIR={log_dir}
export WANDB_CACHE_DIR={log_dir}/.cache/wandb

# 2. PYTHONPATH Setup
# This tells Python to look in the project root for imports
export PYTHONPATH=$PYTHONPATH:{project_root}

# 3. Execution
cd {code_dir}

ENV_PYTHON="{conda_env}/bin/python"

echo "Starting training for {env_name} (Seed: {seed})..."
echo "PYTHONPATH is set to: $PYTHONPATH"

$ENV_PYTHON run.py \\
    --env_name {env_name} \\
    --total_timesteps {timesteps} \\
    --ref_point {ref_point} \\
    --seed {seed}
"""

# ====================================================
# 2. CORE LOGIC
# ====================================================

def main():
    print(f"--- Generating Job Scripts for: {EXPERIMENT_TAG} ---")

    if not JOB_SCRIPTS_DIR.exists():
        JOB_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created job directory: {JOB_SCRIPTS_DIR}")

    submit_commands = []

    # Iterate over Environments
    for env_name, config in ENV_SETTINGS.items():
        
        # Iterate over Seeds
        for seed in SEEDS:
            
            # Create a specific folder structure: env_name/seed_X
            # This ensures logs don't clash
            job_dir = JOB_SCRIPTS_DIR / env_name / f"seed_{seed}"
            job_dir.mkdir(parents=True, exist_ok=True)
            
            script_save_path = job_dir / "submit.sh"
            
            # Format the template
            script_content = SLURM_TEMPLATE.format(
                job_name=f"morld_{env_name}_s{seed}",
                log_dir=job_dir,
                conda_env=CONDA_ENV_PATH,
                code_dir=CODE_DIR,
                project_root=PROJECT_ROOT,  # <--- Add this
                wandb_key=WANDB_API_KEY,
                wandb_entity=WANDB_ENTITY,
                wandb_project=WANDB_PROJECT,
                env_name=env_name,
                timesteps=config["timesteps"],
                ref_point=config["ref_point"],
                seed=seed
            )

            with open(script_save_path, "w") as f:
                f.write(script_content)

            submit_commands.append(f"sbatch {script_save_path}")
            print(f"  Generated: {script_save_path}")

    # Create Master Submit Script
    if submit_commands:
        master_script_path = JOB_SCRIPTS_DIR / f"submit_all_{EXPERIMENT_TAG}.sh"
        with open(master_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Experiment Tag: {EXPERIMENT_TAG}\n")
            f.write("\n".join(submit_commands))
        
        os.chmod(master_script_path, 0o755)

        print("-" * 40)
        print("Generation Complete.")
        print(f"Master submit file: {master_script_path}")
        print("-" * 40)

if __name__ == "__main__":
    main()
