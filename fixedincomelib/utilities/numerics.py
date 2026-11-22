import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """One quadratic per node interval (inspired by Hagan-West's basic quadratic interpolator).

    On [t_i, t_{i+1}] with h = t_{i+1} - t_i, secant s = (f_{i+1} - f_i) / h and y = t - t_i:
        f(t) = f_i + d*y + (s - d) * y^2 / h
    It passes through both node values for any starting slope d, and its slope runs from d to 2s - d.
    - monotone and convex on the interval  <=>  d lies between min(0, 2s) and s
    - d = slope the previous quadratic ends with (smooth join); first interval starts with d = s
    - if that slope is outside the allowed range we clip it, which leaves a small kink at the node
    One node -> constant. FLAT extrapolation outside the nodes.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        self.t = [float(x) for x in axis1]
        self.f = [float(v) for v in values]
        self.s = []       # secant slope of each interval
        self.d = []       # starting slope of each interval's quadratic
        self.cum = [0.0]  # integral of f from t[0] to each node

        for i in range(len(self.t) - 1):
            h = self.t[i + 1] - self.t[i]
            s = (self.f[i + 1] - self.f[i]) / h
            if i == 0:
                d = s                                   # first interval: straight line
            else:
                d = 2 * self.s[i - 1] - self.d[i - 1]   # continue the previous slope
            d = min(max(d, min(0.0, 2 * s)), s)         # keep it monotone and convex
            self.s.append(s)
            self.d.append(d)
            self.cum.append(self.cum[-1] + self._area(i, h))

    def _locate(self, x):
        i = 0
        while i < len(self.t) - 2 and x >= self.t[i + 1]:
            i += 1
        return i

    def _value(self, i, y):
        h = self.t[i + 1] - self.t[i]
        return self.f[i] + self.d[i] * y + (self.s[i] - self.d[i]) * y * y / h

    def _area(self, i, y):
        """Integral of f from t[i] to t[i] + y."""
        h = self.t[i + 1] - self.t[i]
        return self.f[i] * y + self.d[i] * y * y / 2 + (self.s[i] - self.d[i]) * y ** 3 / (3 * h)

    def interpolate(self, x: float) -> float:
        if len(self.t) == 1 or x <= self.t[0]:
            return self.f[0]
        if x >= self.t[-1]:
            return self.f[-1]
        i = self._locate(x)
        return self._value(i, x - self.t[i])

    def _integral_from_start(self, x):
        if len(self.t) == 1 or x <= self.t[0]:
            return self.f[0] * (x - self.t[0])
        if x >= self.t[-1]:
            return self.cum[-1] + self.f[-1] * (x - self.t[-1])
        i = self._locate(x)
        return self.cum[i] + self._area(i, x - self.t[i])

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._integral_from_start(end_x) - self._integral_from_start(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
