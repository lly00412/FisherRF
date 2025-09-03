from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .VC_sel import VCSelector
from .Farthest_sel import FarthestPointSelector
from .V_sel import VarSelector
from .MLP_sel import MLPSelector

methods_dict = {"rand": RandSelector,
                "vcurf":VCSelector,
                "mlp":MLPSelector,
                "H_reg": HRegSelector,
                "variance": VarSelector,
                "Farthest": FarthestPointSelector}

