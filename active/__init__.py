from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .V_sel import VarSelector
from .VC_sel import VCSelector
methods_dict = {"rand": RandSelector, "vcurf":VCSelector, "H_reg": HRegSelector, "variance": VarSelector}


# methods_dict = {"rand": RandSelector, "H_reg": HRegSelector}