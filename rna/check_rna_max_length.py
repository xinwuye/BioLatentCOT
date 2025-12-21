import pandas as pd

file_path = 'extracted/rna_flattened.csv'
chunk_size = 500000  # Process 500k rows at a time
max_len = 0
total_rows = 0

print(f"Scanning {file_path} for maximum sequence length...")

for chunk in pd.read_csv(file_path, usecols=['sequence'], chunksize=chunk_size):
    lengths = chunk['sequence'].astype(str).str.len()
    
    current_max = lengths.max()
    if current_max > max_len:
        max_len = current_max
    
    total_rows += len(chunk)
    print(f"Processed {total_rows} rows... Current Max: {max_len}")

print("-" * 30)
print(f"Final Report:")
print(f"Total Rows: {total_rows}")
print(f"Maximum Sequence Length: {max_len} nucleotides")
print("-" * 30)
