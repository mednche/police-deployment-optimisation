#Instructions to the scheduler
#Run the script with current environment and in the current directory
#$ -V -cwd

#Request some time
#$ -l h_rt=3:00:00

#$ -pe smp 40
#$ -l h_vmem=4G

#Send email at start and end of job
#$ -m be
#$ -M mednche@leeds.ac.uk

#$ -o job.std.out 
#$ -e job.std.err

# conda environment
source activate myenv

python SA_travelSpeed.py
