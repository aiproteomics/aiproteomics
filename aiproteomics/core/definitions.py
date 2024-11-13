from dataclasses import dataclass

# Phospho diagnostic peak
ANNOTATION_pY = "pY"

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
