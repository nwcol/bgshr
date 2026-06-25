#!/bin/bash

bgshr predict_B \
    --Ne 27000 \
    --lookup_tbl data/lookup_tbl_equilibrium.csv.gz \
    --bed data/cds_merged_chr22.bed.gz \
          data/promoters_chr22.bed.gz \
    --rmap data/YRI_recombination_map_hapmap_format_hg38_chr_22.txt.gz \
    --umap data/roulette_tbl_chr22.csv.gz \
    --umap_rate_col avg_mut \
    --shapes 0.215 0.111 \
    --scales 0.0240 0.000500 \
    --p_neus 0.302 0.0 \
    -n 2 \
    --n_cores 8 \
    --chunk_size 200 \
    --spacing 10000 \
    --verbose \
    -o outputs/human_chr22_cds_regulatory_model_landscapes.csv
