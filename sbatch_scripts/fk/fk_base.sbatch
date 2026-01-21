#!/bin/bash
#SBATCH -J base                          # Job name
#SBATCH --output=/home/qw325/log/mfk_%j.out # STDOUT -> /home/qw325/log
#SBATCH --error=/home/qw325/log/mfk_%j.err  # STDERR -> /home/qw325/log
#SBATCH --mail-type=END,FAIL                    # 
#SBATCH --mail-user=qw325@cornell.edu           # 
#SBATCH --nodes=1                               # 节点数
#SBATCH --ntasks=1                              # 任务数
#SBATCH --cpus-per-task=8                    # 每任务 CPU 核数
#SBATCH --mem=24576                             # 每节点内存（MB），按需调整
#SBATCH --time=3-00:00:00                          # 运行时长（3天）
#SBATCH --partition=fang                        # 分区
#SBATCH --gres=gpu:1                            # GPU 数
#SBATCH --chdir=/home/qw325/Project/crossembodimentaction  # 进入项目目录

set -eo pipefail

# 确保日志目录存在（即使上面的 --output/--error 已指向该目录）
mkdir -p /home/qw325/log

# 1) 正确初始化并激活 conda（非交互 shell 必须显式 source）
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate repr

# export PYTHONUNBUFFERED=1

# export CUDA_HOME="$CONDA_PREFIX"

# 4) 启动训练
python /home/qw325/Project/crossembodimentaction/train_fk.py --config-name=config_base
