"""Source model for EccentricTD in bilby PE."""
from pycbc.waveform.utils import taper_timeseries
from pycbc.waveform import get_td_waveform
import numpy as np


def frequency_domain_source_model(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    eccentricity,
    **kwargs
):
    """Frequency domain source model for EccentricTD.

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
    eccentricity: str
        Initial eccentricity
    """
    delta_t = 1/(2 * max(frequency))
    f_min = min(frequency)
    hp_td, hc_td = get_td_waveform(approximant="EccentricTD",
                                   mass1=mass_1,
                                   mass2=mass_2,
                                   eccentricity=eccentricity,
                                   f_ref=f_min,
                                   f_lower=f_min,
                                   delta_t=delta_t,
                                   inclination=theta_jn,
                                   distance=luminosity_distance,
                                   coa_phase=phase)
    hp = taper_timeseries(hp_td,
                          tapermethod='startend')
    hc = taper_timeseries(hc_td,
                          tapermethod='startend')
    df = frequency[1] - frequency[0]
    hpf = hp.to_frequencyseries(delta_f=df)
    hcf = hc.to_frequencyseries(delta_f=df)
    plus = np.interp(frequency, hpf.sample_frequencies, hpf.data)
    cross = np.interp(frequency, hcf.sample_frequencies, hcf.data)
    return {"plus": plus,
            "cross": cross}
