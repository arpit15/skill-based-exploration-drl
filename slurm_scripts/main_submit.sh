#!/bin/bash -f
#SBATCH -p CLUSTER
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16
#SBATCH -t 30:00:00
#SBATCH -J myring

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=arpit15945@gmail.com

## test for openmpi
#mpirun -np 2 singularity exec /home/arpita1/final8.img /home/arpita1/test_ompi/openmpi-2.1.2/ring


mpirun -np 8 singularity exec /home/arpita1/final8.img bash par_myprog.sh
