#!/bin/bash
#SBATCH --job-name=eicu_download
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/asoare/vfl_mlt/logs/eicu_download_%j.out

mkdir -p /home/asoare/vfl_mlt/logs

cd /home/asoare/vfl_mlt

wget -r -N -c -np \
  --user ameliasoare \
  --password "$(cat ~/.physionet_password)" \
  https://physionet.org/files/eicu-crd/2.0/
