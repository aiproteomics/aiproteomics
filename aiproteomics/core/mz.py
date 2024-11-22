from pyteomics import mass

from aiproteomics.core.fragment import Fragment

# Phospho diagnostic peak
MASS_pY = 216.043

# Neutral losses
mass_neutral_loss = {
    "H2O": mass.calculate_mass(formula='H2O'),
    "NH3": mass.calculate_mass(formula='NH3'),
    "H3PO4": mass.calculate_mass(formula='H3PO4')
}

def generate_aa_mass():
    """
        Generates the dict of masses for supported amino acids and modifications.

        Modification compositions are obtained from the Unimod database.
    """

    db = mass.Unimod()
    aa_comp = mass.std_aa_comp.copy()

    # Get relevant modifcations from unimod database
    oxidation = db.by_title("Oxidation")["composition"]
    phospho = db.by_title("Phospho")["composition"]
    carbamidomethyl = db.by_title("Carbamidomethyl")["composition"]
    acetyl = db.by_title("Acetyl")["composition"]

    # Generate modified amino acid compositions
    aa_comp["1"] = aa_comp["M"] + oxidation
    aa_comp["2"] = aa_comp["S"] + phospho
    aa_comp["3"] = aa_comp["T"] + phospho
    aa_comp["4"] = aa_comp["Y"] + phospho
    aa_comp["*"] = acetyl
    aa_comp["C"] = aa_comp["C"] + carbamidomethyl

    # Get masses
    aa_mass = {k: mass.calculate_mass(v) for k, v in aa_comp.items()}

    return aa_mass


aa_mass = generate_aa_mass()


def get_ion_mz(seq, frag: Fragment, aa_mass):
    """
    Calculate the mass of the given fragment for the given sequence.
    """

    ion_type = frag.fragment_type
    ion_break = frag.fragment_series_number
    ion_charge = frag.fragment_charge


    if ion_type[0] in 'abc':
        # If the first entry is acetylation, skip it as not real amino acid (check this!)
        if seq[0] == '*':
            frag_seq = seq[:ion_break+1]
        else:
            frag_seq = seq[:ion_break]
    else:
        frag_seq = seq[-ion_break:]

    return mass.fast_mass(frag_seq, ion_type=ion_type, charge=ion_charge, aa_mass=aa_mass)


def get_precursor_mz(seq, charge, aa_mass):
    """
        Calculates the precursor m/z for the given sequence
    """

    return mass.fast_mass(sequence=seq, charge=charge, aa_mass=aa_mass, ion_type='M')
