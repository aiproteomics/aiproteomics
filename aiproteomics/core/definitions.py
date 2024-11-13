from dataclasses import dataclass

# Phospho diagnostic peak
ANNOTATION_pY = "pY"

# Neutral losses
#mass_neutral_loss = {
#    "H2O": mass.calculate_mass(formula='H2O'),
#    "NH3": mass.calculate_mass(formula='NH3'),
#    "H3PO4": mass.calculate_mass(formula='H3PO4')
#}

# Phospho model mapping
# Supported amino acid modifications
aa_mod_map = {
        'M(UniMod:35)': '1',
        'S(UniMod:21)': '2',
        'T(UniMod:21)': '3',
        'Y(UniMod:21)': '4',
        '(UniMod:1)':   '*',
        'C(UniMod:4)':  'C'
}
