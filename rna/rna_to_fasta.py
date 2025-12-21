import pandas as pd
import os

input_file = 'extracted/rna.csv'
output_file = 'extracted/rna.fasta'

print(f"Reading {input_file}...")

df = pd.read_csv(input_file, usecols=['extracted_sequences'])

print(f"Converting {len(df)} sequences to FASTA format...")

with open(output_file, 'w') as f:
    for idx, row in df.iterrows():
        seq_str = str(row['extracted_sequences'])
        
        clean_seq = seq_str.strip("[]'\" ")
        
        f.write(f">{idx}\n{clean_seq}\n")

print(f"Success! FASTA file saved at: {output_file}")