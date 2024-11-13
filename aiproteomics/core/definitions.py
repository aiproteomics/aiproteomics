# Recognized ion types (not including pY)
ALLOWED_IONS = ['x', 'y', 'z', 'a', 'b', 'c']

# Annotation expected for the phospho diagnostic peak
ANNOTATION_pY = "pY"

# Mapping from amino acid to integers in the model input layer
# Note that this includes amino acids with modifications.
aa_int_map = {
        'A': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'P': 13,
        'Q': 14,
        'R': 15,
        'S': 16,
        'T': 17,
        'V': 18,
        'W': 19,
        'Y': 20,
        '1': 21,
        '2': 22,
        '3': 23,
        '4': 24,
        '*': 25
    }

# Supported amino acid modifications for the phospho model
# and mapping to its alphabet
aa_mod_map = {
        'M(UniMod:35)': '1',
        'S(UniMod:21)': '2',
        'T(UniMod:21)': '3',
        'Y(UniMod:21)': '4',
        '(UniMod:1)':   '*',
        'C(UniMod:4)':  'C'
}
