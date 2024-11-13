from dataclasses import dataclass
import itertools as it

@dataclass
class Fragment:
    fragment_type: str
    fragment_charge: int
    fragment_series_number: int
    fragment_loss_type: str

    def __post_init__(self):
        self.annotation = self.make_annotation_str()

    def make_annotation_str(self) -> str:
        s = self.fragment_type + str(self.fragment_series_number)
        if self.fragment_charge > 1:
            s += f'({str(self.fragment_charge)}+)'
        if self.fragment_loss_type:
            s += f'-{self.fragment_loss_type}'
        return s
