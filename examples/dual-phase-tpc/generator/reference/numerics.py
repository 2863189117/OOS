"""Numerical helpers used by the LXe generator."""
from __future__ import annotations
import math
import numpy as np

def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isinf(number):
            return "infinity" if number > 0 else "-infinity"
        if math.isnan(number):
            raise ValueError("NaN cannot be serialized")
        return number
    if isinstance(value, np.integer):
        return int(value)
    return value

def gauss_interval(order: int, lower: float, upper: float):
    if order <= 0:
        raise ValueError("Gauss-Legendre order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scale = 0.5 * (upper - lower)
    return lower + scale * (nodes + 1.0), scale * weights

def fresnel_power(n_incident: float, n_transmitted: float, cos_incident: float):
    ci = float(np.clip(cos_incident, 0.0, 1.0))
    sin2_i = max(0.0, 1.0 - ci * ci)
    sin2_t = (n_incident / n_transmitted) ** 2 * sin2_i
    if sin2_t >= 1.0:
        return np.ones(2), np.zeros(2), None
    ct = math.sqrt(max(0.0, 1.0 - sin2_t))
    rs = (n_incident * ci - n_transmitted * ct) / (n_incident * ci + n_transmitted * ct)
    rp = (n_transmitted * ci - n_incident * ct) / (n_transmitted * ci + n_incident * ct)
    reflectance = np.asarray([rs * rs, rp * rp], dtype=float)
    return reflectance, 1.0 - reflectance, ct
