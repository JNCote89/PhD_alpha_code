import numpy as np
import scipy.stats as st


def t_confidence_interval(data: np.array, confidence: float = 0.95, precision: int = 3) -> tuple[float, float]:
    low, high = st.t.interval(confidence=confidence, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    return round(low, precision), round(high, precision)


def z_confidence_interval(data: np.array, confidence: float = 0.95, precision: int = 3) -> tuple[float, float]:
    low, high = st.norm.interval(confidence=confidence, loc=np.mean(data), scale=st.sem(data))
    return round(low, precision), round(high, precision)
