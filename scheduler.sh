#!/bin/bash
for s in 5 6 7 8 9; do
        for d in barlow_twins.sh byol.sh vicreg.sh simclr.sh mocov2plus.sh ; do
            >batch_script.slurm
            echo "#!/bin/bash" >> batch_script.slurm
            echo "#SBATCH --job-name="$d"_"$s"_normal_transforms" >> batch_script.slurm
            echo "#SBATCH --ntasks=1" >> batch_script.slurm
            echo "#SBATCH --gres=gpu:1 " >> batch_script.slurm
            echo "#SBATCH -A kio@v100 " >> batch_script.slurm
            echo "#SBATCH -C v100-32g" >> batch_script.slurm
            echo "#SBATCH --qos=qos_gpu-t3" >> batch_script.slurm
            echo "#SBATCH --cpus-per-task=6" >> batch_script.slurm
            echo "#SBATCH --hint=nomultithread" >> batch_script.slurm
            echo "#SBATCH --time=16:40:00" >> batch_script.slurm
            echo "#SBATCH --output=./terminal_normal_transforms/"$d"_"$s"_normaltransforms%j.out " >> batch_script.slurm
            echo "#SBATCH --error=./errors_normal_transforms/"$d"_"$s"_normaltransforms%j.out" >> batch_script.slurm
            echo "module load pytorch-gpu/py3/1.7.1" >> batch_script.slurm
            echo "conda deactivate" >> batch_script.slurm
            echo "conda activate clean" >> batch_script.slurm
            echo "nvidia-smi" >> batch_script.slurm
            echo "set -x" >> batch_script.slurm
            echo "bash bash_files/cifar100/resnet18/"$d" "$v $s >> batch_script.slurm
            sbatch batch_script.slurm 
        done
    done
done