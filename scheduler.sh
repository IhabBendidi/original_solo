#!/bin/bash

for v in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5  0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 0.55 0.45 0.65 0.35 0.675 0.625 0.325 0.375 0.475 0.425 0.525 0.575; do
    for d in barlow_twins.sh simclr.sh vicreg.sh mocov2plus.sh byol.sh deepclusterv2.sh nnbyol.sh nnclr.sh ressl.sh simsiam.sh swav.sh vibcreg.sh; do
        >batch_script.slurm
        echo "#!/bin/bash" >> batch_script.slurm
        echo "#SBATCH --job-name="$d"_"$v"_jitter" >> batch_script.slurm
        echo "#SBATCH --ntasks=1" >> batch_script.slurm
        echo "#SBATCH --gres=gpu:1 " >> batch_script.slurm
        echo "#SBATCH -C v100-32g" >> batch_script.slurm
        echo "#SBATCH --qos=qos_gpu-t3" >> batch_script.slurm
        echo "#SBATCH --cpus-per-task=6" >> batch_script.slurm
        echo "#SBATCH --hint=nomultithread" >> batch_script.slurm
        echo "#SBATCH --time=02:40:00" >> batch_script.slurm
        echo "#SBATCH --output=./terminal/"$d"_"$v"_jitter%j.out " >> batch_script.slurm
        echo "#SBATCH --error=./errors/"$d"_"$v"_jitter%j.out" >> batch_script.slurm
        echo "module load pytorch-gpu/py3/1.7.1" >> batch_script.slurm
        echo "conda deactivate" >> batch_script.slurm
        echo "conda activate solo" >> batch_script.slurm
        echo "nvidia-smi" >> batch_script.slurm
        echo "set -x" >> batch_script.slurm
        echo "bash bash_files/cifar10/"$d" "$v >> batch_script.slurm
        sbatch batch_script.slurm 
    done
done