from pydantic import BaseModel
from typing import Literal


class ParametersConfig(BaseModel):
    fitting_methods: list[Literal["aswift", "poly_linear"]]
    hz_values: list[int]
    calibration_lower_idx: int
    calibration_upper_idx: int
    bg_buffer: int = 0
    baseline_boundary: float = 0.05
    huber_reweight: bool = True
    use_file0: bool = True
    full_cutoff: float = 1.
    peak_lambda_scale: float = 1.0
    peak_prominence: float = 0.5
    huber_cutoff: float = 2.0
    mad_window: int = 21


class LocationsConfig(BaseModel):
    input_dir: 'str'


class Config(BaseModel):
    parameters: ParametersConfig
    locations: LocationsConfig
