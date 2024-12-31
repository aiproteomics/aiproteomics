from dataclasses import dataclass

@dataclass
class Fragment:
    """
    Stores the details for an individual peptide fragment:
        `fragment_type`: y, b, a, etc. Possible values can be found in `aiproteomics.core.definitions`
        `fragment_charge`: The integer charge on the ion, must be greater than 0
        `fragment_series_number`: The position in the precursor sequence where the breakage occured
        `fragment_loss_type`: What loss occured (if any) i.e. "H2O", "NH3", "H3PO4", or empty string
    """

    fragment_type: str
    fragment_charge: int
    fragment_series_number: int
    fragment_loss_type: str

    def __post_init__(self):
        """
        For reasons of convenience (and performance), pre-calculate and store the
        string representation of this fragment.
        """

        self.annotation = self.make_annotation_str()

    def make_annotation_str(self) -> str:
        """
        Returns the string representation of this peptide fragment.
        """

        s = self.fragment_type + str(self.fragment_series_number)
        if self.fragment_charge > 1:
            s += f'({str(self.fragment_charge)}+)'
        if self.fragment_loss_type:
            s += f'-{self.fragment_loss_type}'
        return s
