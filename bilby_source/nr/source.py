"""Source models for Numerical Relativity waveform injection in bilby PE."""
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform.utils import taper_timeseries
import h5py
import lal
import numpy as np


def frequency_domain_nr_source_model(
    frequecny,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    data_file
):
    """Frequency domain source model for injecting NR data.

    Parameters:
    -----------
    mass_1: float
        First component mass in units of solar mass to scale the NR waveform.
    mass_2: float
        Second component mass in units of solar mass to scale the NR waveform.
    luminosity_distance: float
        Luminosity distance in units of Mpc to scale the NR waveform.
    theta_jn: float
        Inclination angle to compute sYlm.
    phase: float
        phase to compute the sYlm.
    data_file: str
        Path to the NR data file.
    """
    d = h5py.File(data_file, "r")
    m_total = mass_1 + mass_2
    t = d["t"][:] * lal.MTSUN_SI * m_total
    t = t - t[0]
    h22 = d["h22"][:]
    dt = 1/(2 * max(frequecny))
    time = np.arange(t[0], t[-1], dt)
    h22_interp = (np.interp(time, t, h22)
                  * (1/(luminosity_distance * 10**6 * lal.PC_SI))
                  * (lal.C_SI * m_total * lal.MTSUN_SI))
    h = lal.SpinWeightedSphericalHarmonic(
        theta_jn, phase, -2, 2, 2) * h22_interp
    hp = taper_timeseries(TimeSeries(np.real(h), delta_t=dt),
                          tapermethod='startend')
    hc = taper_timeseries(TimeSeries(- np.imag(h), delta_t=dt),
                          tapermethod='startend')
    hpf = hp.to_frequencyseries()
    hcf = hc.to_frequencyseries()
    plus = np.interp(frequecny, hpf.sample_frequencies, hpf.data)
    cross = np.interp(frequecny, hcf.sample_frequencies, hcf.data)
    return {"plus": plus,
            "cross": cross}
