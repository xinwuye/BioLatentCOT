import pandas as pd 
import json
import re 

def extract_input_to_df(df, line, seq):
    """
    Appends a JSONL line and its extracted sequence to a DataFrame.
    """
    data = json.loads(line)
    
    data['extracted_sequences'] = seq
    
    new_row_df = pd.DataFrame([data])
    
    if df is None or df.empty:
        return new_row_df
    else:
        return pd.concat([df, new_row_df], ignore_index=True)

file_path = 'stage2_train.jsonl'
rna = None # each element is a list containing rna sequences appeared in the input 
protein = None
dna = None 
other_tags = set()
pattern_rna = r"<rna>(.*?)<rna>"
pattern_dna = r"<dna>(.*?)<dna>"
pattern_protein = r"<protein>(.*?)<protein>"
other = r"<(.*?)>"

with open(file_path, 'r') as file:
    length = 3330232 #sum(1 for line in file)
    iter = 0
    print(length)
    for line in file:
        # print(iter/length)
        if iter % 100000 == 0: 
            print(iter/length)
        data = json.loads(line)
        # print(data)
        input = data['input']
        # print(input)
        iter += 1

        others = re.findall(other, input) 
        for tag in others: 
            if (tag != "protein") and (tag != "rna") and (tag != "dna"):
                other_tags.append(tag)

        rna_seq = re.findall(pattern_rna, input) # list
        if rna_seq: 
            dna_seq = re.findall(pattern_dna, input)
            if dna_seq: # rna + dna case
                continue 
            protein_seq = re.findall(pattern_protein, input) 
            if protein_seq: # rna + protein case 
                continue 
        rna = extract_input_to_df(rna, line, rna_seq)
        continue 

        dna_seq = re.findall(pattern_dna, input)
        if dna_seq: 
            protein_seq = re.findall(pattern_protein, input) 
            if protein_seq:
                continue # dna + protein case
        dna = extract_input_to_df(dna, line, dna_seq)
        continue 

        protein_seq = re.findall(pattern_protein, input)
        protein = extract_input_to_df(protein, line, protein_seq)

        # <rna_sequence_here> .... <protein_sequence_here> ''' embedding, is there stay the same, special token, replace by model embedding from rna foundation model" ''' 
        # only single molecule task 
        # check rna_protein_interaction or other multi-one -> get rid of them 
        # tokenize insert special token <rsq>  
        
print(other_tags)
output_dir = './extracted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rna.to_csv(os.path.join(output_dir, 'rna.csv'), index=False)
dna.to_csv(os.path.join(output_dir, 'dna.csv'), index=False)
protein.to_csv(os.path.join(output_dir, 'protein.csv'), index=False)

print(f"Files saved successfully in {output_dir}")