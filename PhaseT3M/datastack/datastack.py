from typing import Optional,Sequence,Union
import numpy as np

class DataStack():

    def __init__(
        self,
        data: np.ndarray,
        defocus: Sequence[float],
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
        self.name = name
        self.units = units


        
