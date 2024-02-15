#!/bin/bash
#SBATCH --job-name=run_genome_wide_locke
#SBATCH --time=40:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180GB
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aprilkim@jhu.edu
#SBATCH -e prsfnn_full_locke_thresh0.1_stdinfo.txt
#SBATCH -o prsfnn_full_locke_thresh0.1_stdout.txt


for chrom_file in data/ld_blocks/*; do
    (
        # Process each region in the chromosome file serially
        while IFS= read -r region; do
		ld_block_name=$region
		echo "$ld_block_name"
		~/data-abattle4/april/programs/julia-1.9.4/bin/julia --project=. -t 2 -e 'using Revise, PRSFNN; PRSFNN.main(ARGS[1], "/data/abattle4/april/hi_julia/annotations/ccre/celltypes", "/data/abattle4/jweins17/LD_REF_PANEL/output/bcf")' "$ld_block_name"
        done < "$chrom_file"
    ) &
done

# Wait for all background processes (chromosomes) to finish
wait

