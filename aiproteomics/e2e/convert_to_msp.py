# Code from (or adapted from) https://github.com/kusterlab/prosit/ Apache License 2.0
# See README.md

import numpy as np
import pyteomics
from pyteomics import mass  # pylint: disable=unused-import

from aiproteomics.e2e import constants
from aiproteomics.e2e import utils


def generate_aa_comp():
    """
    >>> aa_comp = generate_aa_comp()
    >>> aa_comp["M"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 1, 'N': 1})
    >>> aa_comp["Z"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 2, 'N': 1})
    """

    db = pyteomics.mass.mass.Unimod()
    aa_comp = dict(pyteomics.mass.std_aa_comp)
    s = db.by_title("Oxidation")["composition"]
    aa_comp["Z"] = aa_comp["M"] + s
    s = db.by_title("Carbamidomethyl")["composition"]
    aa_comp["C"] = aa_comp["C"] + s
    return aa_comp


aa_comp = generate_aa_comp()


def get_ions():
    x = np.empty(
        [constants.MAX_ION, len(constants.ION_TYPES), constants.MAX_FRAG_CHARGE],
        dtype="|S6",
    )
    for fz in range(constants.MAX_FRAG_CHARGE):
        for fty_i, fty in enumerate(constants.ION_TYPES):
            for fi in range(constants.MAX_ION):
                ion = fty + str(fi + 1)
                if fz > 0:
                    ion += "({}+)".format(fz + 1)
                x[fi, fty_i, fz] = ion
    x.flatten()
    return x


ox_int = constants.ALPHABET["M(ox)"]
c_int = constants.ALPHABET["C"]


def calculate_mods(sequence_integer):
    """
    >>> x = np.array([2, 15, 4, 3, 0, 0])
    >>> calculate_mods(x)
    1
    >>> x = np.array([2, 15, 21, 3, 0, 0])
    >>> calculate_mods(x)
    2
    """
    return len(np.where((sequence_integer == ox_int) | (sequence_integer == c_int))[0])


def generate_mods_string_tuples(sequence_integer):
    list_mods = []
    for mod in [ox_int, c_int]:
        for position in np.where(sequence_integer == mod)[0]:
            if mod == c_int:
                list_mods.append((position, "C", "Carbamidomethyl"))
            elif mod == ox_int:
                list_mods.append((position, "M", "Oxidation"))
            else:
                raise ValueError("cant be true")
    list_mods.sort(key=lambda tup: tup[0])  # inplace
    return list_mods


def generate_mod_strings(sequence_integer):
    """
    >>> x = np.array([1,2,3,1,2,21,0])
    >>> y, z = generate_mod_strings(x)
    >>> y
    '3/1,C,Carbamidomethyl/4,C,Carbamidomethyl/5,M,Oxidation'
    >>> z
    'Carbamidomethyl@C2; Carbamidomethyl@C5; Oxidation@M6'
    """
    list_mods = generate_mods_string_tuples(sequence_integer)
    if len(list_mods) == 0:
        return "0", ""

    returnString_mods = ""
    returnString_modString = ""
    returnString_mods += str(len(list_mods))
    for i, mod_tuple in enumerate(list_mods):
        returnString_mods += "/" + str(mod_tuple[0]) + "," + mod_tuple[1] + "," + mod_tuple[2]
        if i == 0:
            returnString_modString += mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
        else:
            returnString_modString += (
                "; " + mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
            )

    return returnString_mods, returnString_modString


def convert_to_speclib(data, fmt='tsv', intensity_scaling=10000, unknown_value_str='NA'):
    """
    Convert the given data frame of prediction data into a speclib format.
    Supported formats are currently 'msp' and 'tsv'
    """

    IONS = get_ions().reshape(174, -1).flatten()

    speclibtxt = ''
    for i in range(data["iRT"].shape[0]):
        # Get the intensity, ion and mass for all positive intensities
        aIntensity = data["intensities_pred"][i]
        sel = np.where(aIntensity > 0)
        aIntensity = aIntensity[sel]
        aIons = IONS[sel]
        aMass = data["masses_pred"][i][sel]

        # Get the original inputs to the fragmentation model
        collision_energy = data["collision_energy_aligned_normed"][i] * 100
        precursor_charge = data["precursor_charge_onehot"][i].argmax() + 1
        sequence_integer = data["sequence_integer"][i]

        # Get the predicted normalized retention time and ion mobility
        # for this sequence (if available)
        if "iRT" in data:
            iRT = np.squeeze(data["iRT"][i])
        else:
            iRT = None

        if "ccs" in data:
            ccs = np.squeeze(data["ccs"][i])
        else:
            ccs = None

        # If available, use the protein id and gene name in the spec lib
        protein_id = data["protein_id"][i]
        gene_name = data["gene_name"][i]

        # Build a predicted Spectrum representation for this sequence
        spec = Spectrum(
            aIntensity,
            collision_energy,
            iRT,
            ccs,
            aMass,
            precursor_charge,
            sequence_integer,
            aIons,
            protein_id,
            gene_name,
            intensity_scaling,
            unknown_value_str
        )

        # Output string representation of the spectrum in the chosen format
        if fmt == 'msp':
            speclibtxt += spec.to_msp()
        elif fmt == 'tsv':
            speclibtxt += spec.to_tsv()
        else:
            raise ValueError('Format not supported. Supported formats are msp and tsv')

    return speclibtxt


class Spectrum:
    def __init__(
        self,
        aIntensity,
        collision_energy,
        iRT,
        ccs,
        aMass,
        precursor_charge,
        sequence_integer,
        aIons,
        protein_id,
        gene_name,
        intensity_scaling,
        unknown_value_str='NA'
    ):
        self.intensity_scaling = intensity_scaling

        self.aIntensity = aIntensity
        self.collision_energy = collision_energy

        if iRT:
            self.iRT = str(iRT)
        else:
            self.iRT = unknown_value_str

        if ccs:
            self.ccs = str(ccs)
        else:
            self.ccs = unknown_value_str

        self.aMass = aMass
        self.precursor_charge = precursor_charge
        self.aIons = aIons
        self.mod, self.mod_string = generate_mod_strings(sequence_integer)
        self.sequence = utils.get_sequence(sequence_integer)
        # amino acid Z which is defined at the toplevel in generate_aa_comp
        self.precursor_mass = pyteomics.mass.calculate_mass(
            self.sequence.replace("M(ox)", "Z"),
            aa_comp=aa_comp,
            ion_type="M",
            charge=int(self.precursor_charge),
        )
        self.protein_id = protein_id
        self.gene_name = gene_name


    def to_msp(self):
        s = "Name: {sequence}/{charge}\nMW: {precursor_mass}\n"
        s += "Comment: Parent={precursor_mass} Collision_energy={collision_energy} "
        s += "Mods={mod} ModString={sequence}//{mod_string}/{charge} "
        s += "iRT={iRT}"
        s += "ccs={ccs}"
        s += "\nNum peaks: {num_peaks}"
        num_peaks = len(self.aIntensity)
        s = s.format(
            sequence=self.sequence.replace("M(ox)", "M"),
            charge=self.precursor_charge,
            precursor_mass=self.precursor_mass,
            collision_energy=np.round(self.collision_energy[0], 0),
            mod=self.mod,
            mod_string=self.mod_string,
            iRT=self.iRT,
            num_peaks=num_peaks,
        )
        for mz, intensity, ion in zip(self.aMass, self.aIntensity, self.aIons):
            s += "\n" + str(mz) + "\t" + str(intensity) + '\t"'
            s += ion.decode("UTF-8").replace("(", "^").replace("+", "") + '/0.0ppm"'
        return s + '\n'


    def to_tsv(self):
        # Remove oxidation notation from sequence string
        peptide_seq = self.sequence.replace("M(ox)", "M")

        s = ''

        # Add rows for each fragment
        for mz, intensity, ion in zip(self.aMass, self.aIntensity, self.aIons):
            # TODO: Make this values->string->values more sane
            ion_str = ion.decode('UTF-8')
            fragment_type = ion_str[0]
            fragment_series_number = ion_str.split('(')[0][1:]
            if '(' in ion_str:
                fragment_charge = ion_str.split('(')[1].split('+')[0] # This is horrible
            else:
                fragment_charge = '1'
            annotation = fragment_type + fragment_series_number + '^' + fragment_charge

            # Scale intensity (often the highest peak is 10000 by convention)
            intensity *= self.intensity_scaling

            # Collect the values for each column describing a particular peak
            row = [
                str(self.precursor_mass), 
                str(mz),
                annotation,
                str(self.protein_id),
                str(self.gene_name),
                peptide_seq,
                self.sequence,
                str(self.precursor_charge),
                str(intensity),
                self.iRT,
                self.ccs,
                fragment_type,
                fragment_charge,
                fragment_series_number,
                'NA'
            ]
            # Add the row to the output string
            s += '\t'.join(row) + '\n'
            
        return s

def get_tsv_format_header():
    tsv_columns = [
        "PrecursorMz",
        "ProductMz",
        "Annotation",
        "ProteinId",
        "GeneName",
        "PeptideSequence",
        "ModifiedPeptideSequence",
        "PrecursorCharge",
        "LibraryIntensity",
        "NormalizedRetentionTime",
        "PrecursorIonMobility",
        "FragmentType",
        "FragmentCharge",
        "FragmentSeriesNumber",
        "FragmentLossType"
    ]
    return '\t'.join(tsv_columns) + '\n'

