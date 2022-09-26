#!/bin/bash
for s in 5 6 7; do
    for v in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        for d in byol.sh vicreg.sh simclr.sh dino.sh ; do
            >batch_script.slurm
            echo "#!/bin/bash" >> batch_script.slurm
            echo "#SBATCH --job-name="$d"_"$s"_"$v"_jitter" >> batch_script.slurm
            echo "#SBATCH --ntasks=1" >> batch_script.slurm
            echo "#SBATCH --gres=gpu:1 " >> batch_script.slurm
            echo "#SBATCH -C v100-32g" >> batch_script.slurm
            echo "#SBATCH --qos=qos_gpu-t3" >> batch_script.slurm
            echo "#SBATCH --cpus-per-task=6" >> batch_script.slurm
            echo "#SBATCH --hint=nomultithread" >> batch_script.slurm
            echo "#SBATCH --time=16:40:00" >> batch_script.slurm
            echo "#SBATCH --output=./terminal/"$d"_"$s"_"$v"_hue%j.out " >> batch_script.slurm
            echo "#SBATCH --error=./errors/"$d"_"$s"_"$v"_hue%j.out" >> batch_script.slurm
            echo "module load pytorch-gpu/py3/1.7.1" >> batch_script.slurm
            echo "conda deactivate" >> batch_script.slurm
            echo "conda activate clean" >> batch_script.slurm
            echo "nvidia-smi" >> batch_script.slurm
            echo "set -x" >> batch_script.slurm
            echo "bash bash_files/imagenet100/"$d" "$v $s >> batch_script.slurm
            sbatch batch_script.slurm 
        done
    done
done