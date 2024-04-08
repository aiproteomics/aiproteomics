import csv
from aiproteomics.e2e import constants

def sequence_has_valid_alphabet(seq, alpha):
    for char in seq:
        if char not in alpha:
            return False
    return True


def extract_sequences(
        tsv_fname,
        out_csv_fname,
        all_peptides_column='All.Peptides',
        protein_id_column='Uniprot.ID',
        gene_name_column='Gene.Name',
        min_peptide_len=5,
        max_peptide_len=30,
        charges=None,
        collision_energies=None,
        allowed_alphabet=None):

    if charges is None:
        charges = [1,2,3]

    if collision_energies is None:
        collision_energies = [20]

    if allowed_alphabet is None:
        allowed_alphabet=constants.ALPHABET

    gene_name_map = {}
    protein_id_map = {}

    with open(tsv_fname, "r", encoding="utf8") as infile:
        tsv_reader = csv.reader(infile, delimiter="\t")

        header = next(tsv_reader)

        # Get index/columnheading mapping
        col_index = { k:v for v, k in enumerate(header)}

        # Create the set of all peptides
        peptides_set = set()
        for row in tsv_reader:
            all_peptides_list = row[col_index[all_peptides_column]].split(';')
            filter_allowed_sequence_lengths = filter (
                    lambda pep_seq: (len(pep_seq) >= min_peptide_len) and (len(pep_seq) <= max_peptide_len), all_peptides_list)
            peptides_set.update(filter_allowed_sequence_lengths)

            # Build mapping of protein sequences to one or more gene names and protein ids
            for peptide in all_peptides_list:
                if peptide not in gene_name_map:
                    gene_name_map[peptide] = set()
                if peptide not in protein_id_map:
                    protein_id_map[peptide] = set()
                gene_name_map[peptide].update(row[col_index[gene_name_column]].split(';'))
                protein_id_map[peptide].update(row[col_index[protein_id_column]].split(';'))

    # Filter for allowed alphabet
    allowed_peptides_set = {pep for pep in peptides_set if sequence_has_valid_alphabet(pep, allowed_alphabet)}

    with open(out_csv_fname, "w") as outfile:
        # Write header
        outfile.write("protein_id,gene_name,modified_sequence,collision_energy,precursor_charge\n")

        # Write sequences with various charges and collision energies
        for seq in allowed_peptides_set:
            protein_id = ";".join(protein_id_map[seq])
            gene_name = ";".join(gene_name_map[seq])
            for collision_energy in collision_energies:
                for precursor_charge in charges:
                    outfile.write(f'{protein_id},{gene_name},{seq},{collision_energy},{precursor_charge}\n')

