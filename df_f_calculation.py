import numpy as np
import pandas as pd


def dff_calc(
    data: np.ndarray,
    fps: float = 30.0,
    tau_0: float = 0.1,
    tau_1: float = 0.35,
    tau_2: float = 2.0,
    invert: bool = False,
    low_background_flag: bool = False,
):
    """Calculates dF/F from calcium traces, based on
    https://www.nature.com/articles/nprot.2010.169.

    Takes a matrix with rows as independent fluorescent traces
    and returns a similar-sized matrix with the corresponding dF/F values
    calculated based on https://www.nature.com/articles/nprot.2010.169.
    The starting frames might have a 0 dF/F value, due to NaNs being
    converted to 0. This happens following the running window operations
    that are preformed on the data.

    Parameters
    ----------
    data : np.ndarray
        dF/F traces with dimensions (cell x time)
    fps : float, optional
        Frame rate (Hz)
    tau_0 : float, optional
        Exponential smoothing factor in seconds
    tau_1 : float, optional
        F0 smoothing parameter in seconds
    tau_2 : float, optional
        Time window before each measurement to minimize
    invert : bool, optional
        False (default) if the transient is expected to be positive, True
        otherwise.
    low_background_flag : bool, optional
        False (default) if the baseline is strictly positive, as in analog imaging. True for low background recordings, e.g. photon counting, three-photon excitation or sparse acquisition.

    Returns
    -------
    dff : np.ndarray
        A 2D array, each row being a calculated dF/F trace.
    """
    low_background_offset_value = 0.05
    tau_0, tau_1, tau_2, min_per = _apply_units_and_corrections(
        fps, tau_0, tau_1, tau_2
    )
    if invert:
        data = -data

    f0 = _calc_f0(data, tau_1, tau_2, min_per)
    unfiltered_dff = _calc_dff_unfiltered(f0, data, low_background_flag, low_background_offset_value)
    dff = _filter_dff(unfiltered_dff, tau_0, min_per)
    if data.ndim == 1:
        return np.atleast_2d(dff.to_numpy().ravel())
    else:
        return dff.to_numpy().T


def _apply_units_and_corrections(fps, tau_0, tau_1, tau_2):
    """Correct the given parameters based on the FPS"""
    tau_0 = fps * tau_0
    tau_1 = int(fps * tau_1)
    tau_2 = int(fps * tau_2)
    min_per = max(1, int(fps / 10))
    return tau_0, tau_1, tau_2, min_per


def _calc_f0(
    data: np.ndarray, tau_1: float, tau_2: float, min_per: int
) -> pd.DataFrame:
    """Create the F_0(t) baseline for the dF/F calculation using a boxcar window."""
    data = pd.DataFrame(data.T)
    f0 = (
        data.rolling(window=tau_1, win_type="boxcar")
        .mean()
        .rolling(window=tau_2, min_periods=min_per)
        .min()
        + np.finfo(float).eps
    )

    return f0


def _calc_dff_unfiltered(f0: pd.DataFrame, data: np.ndarray, low_background_flag: bool, low_background_offset_value=0.0):
    """ Subtract baseline from current fluorescence """
    f0 = f0.to_numpy()
    if low_background_flag:
        raw_calc = (data.T  - f0) / (f0 + low_background_offset_value) # artificially increasing the baseline for low-background imaging modalities, so as to prevent unrealistically high dF/F values.
    else:
        raw_calc = (data.T - f0) / f0
    unfiltered_dff = pd.DataFrame(raw_calc).fillna(0)
    return unfiltered_dff


def _filter_dff(unfiltered_dff: pd.DataFrame, tau_0: float, min_per: int):
    """Apply an exponentially weighted moving average to the dF/F data. """
    dff = unfiltered_dff.ewm(halflife=tau_0, min_periods=min_per).mean().fillna(0)
    return dff

