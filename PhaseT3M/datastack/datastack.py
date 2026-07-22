from typing import Optional,Sequence,Union
import numpy as np

class DataStack():

    def __init__(
        self,
        data: np.ndarray,
        defocus: Sequence[float],
        A1x: Optional[Sequence[float]] = None,
        A1y: Optional[Sequence[float]] = None,
        name: Optional[str] = 'A focal series of 2D_array',
        units: Optional[str] = '',
        dims: Optional[list] = None,
        dim_names: Optional[list] = None,
        dim_units: Optional[list] = None,
        #slicelabels = None
        ):

        #super().__init__()

        # array & metadata
        self.data = data
        self.defocus = defocus
        # two-fold astigmatism (C12a, C12b), same convention as PhaseT3M's
        # aberrations_basis_function coefficients -> optional, defaults to 0
        self.A1x = A1x
        self.A1y = A1y
        self.name = name
        self.units = units


        
