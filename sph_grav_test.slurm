#!/bin/bash
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --exclusive
#SBATCH -J immerTQM.slurm
#SBATCH -t 00:30:00
#SBATCH --output=sph_grav_test.slurm.out

cd /scratch/gpfs/yc9993/athenaGI

RUNDIR=test

if [ -d $RUNDIR ] ; then
    echo "directory exists"
else
    mkdir -p $RUNDIR
fi

cd $RUNDIR

#cp ../vis/python/athena_read.py athena_read.py


module load anaconda3/2020.11 gcc-toolset/10 openmpi/gcc/4.1.0  fftw/gcc/openmpi-4.1.0/3.3.9 hdf5/gcc/openmpi-4.1.0/1.10.6  eigen3/3.3.4

# make sure to load same modules used to compile the code
#module load intel-mpi/gcc/2021.3.1 fftw/gcc/intel-mpi/3.3.9 hdf5/gcc/intel-mpi/1.10.6
#module load intel/2021.1.2 intel-mpi/intel/2021.1.1 fftw/intel-2021.1/3.3.9 hdf5/intel-2021.1/intel-mpi/1.10.6
#run the application:

#srun -n 1120 ../bin/athena  -t 23:55:00 -r rh300disk_pole.final.rst  -i ../inputs/athinput.rh300disk_T6e4_rho1e-8
#srun -n 480 ../bin/athena  -t 23:50:00 -r disk.initial.rst  -i ../inputs/athinput.AGN_newunit
srun -n 8  ../bin/athena  -t 23:50:00   -i  ../my_tests/test_sph_gravity.in

