import os
from Bio import PDB
from Bio.SeqUtils import seq1
from tqdm import tqdm

def extract_first_chain_to_fasta(input_folder, output_fasta):
    # Initialize the PDB parser and suppress warnings.
    parser = PDB.PDBParser(QUIET=True)
    
    with open(output_fasta, 'w') as fasta_out:
        # Iterate over all files in the directory.
        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith(".pdb"):
                file_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                
                # Parse the first chain ID from a filename such as pdbid_A_B_C.
                # Assume the format is always pdbid_Chain1_Chain2...
                parts = base_name.split('_')
                if len(parts) < 2:
                    print(f"Skipping {filename}: unexpected filename format")
                    continue
                
                target_chain_id = parts[1]
                
                try:
                    # Parse the PDB structure.
                    structure = parser.get_structure(base_name, file_path)
                    
                    # Get the first model; PDB files usually contain only one.
                    model = structure[0]
                    
                    if target_chain_id in model:
                        chain = model[target_chain_id]
                        
                        # Extract the residue sequence.
                        amino_acids = []
                        for residue in chain:
                            amino_acids.append(residue.get_resname())
                        
                        # Convert three-letter residue codes to one-letter codes.
                        sequence = seq1(''.join(amino_acids))
                        
                        if sequence:
                            fasta_out.write(f">{base_name}\n")
                            fasta_out.write(f"{sequence}\n")
                            #print(f"Successfully extracted {filename} (chain {target_chain_id})")
                        else:
                            print(f"Warning: no valid amino-acid sequence found for chain {target_chain_id} in {filename}")
                    else:
                        print(f"Error: chain {target_chain_id} not found in {filename}")
                        
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")

if __name__ == "__main__":
    # Configure the input directory and output filename.
    input_dir = "/home/kechen/peppocketgen/dataset/propedia/merged_complex"  
    output_file = "../../dataset/propedia_pep_sequences.fasta"
    
    extract_first_chain_to_fasta(input_dir, output_file)
    print(f"\nAll sequences saved to: {output_file}")