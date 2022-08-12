import numpy as np
from numpy.polynomial import Polynomial as Poly
import pandas as pd

from jupyterplot import ProgressPlot as PP

### Same script as lfp_simulate but the simulate function is changed
### to accomodate a ProgressPlot object

def generate_ocv_curve(ocv: np.ndarray):
    '''
    numpy.polynomial.Polynomial.fit
    generates soc -> ocv mapping polynomial of degree 8
    and outputs ocv values at all unit soc values from 100% to 10%
    '''
    assert isinstance(ocv, np.ndarray)
    soc = np.arange(100,0,-10)
    assert len(ocv) == len(soc)
    assert len(ocv) == 10
    curve = Poly.fit(soc, ocv, 8, domain = [5,100])
    return curve.linspace(96, domain = (100, 5))

# Second Order RC-Model
def model_2rc(current, delta_t, u_rc, ocv, r_int, r, c):
    # returns the new voltage and polarization voltage
    tau_i = r * c
    u_rc = np.exp(-delta_t / tau_i) * u_rc + r * \
        (1 - np.exp(-delta_t / tau_i)) * (-current)

    return ocv - r_int * abs(current) - u_rc.sum(), u_rc


def lfp_cell(capacity: float, delta_t: float,
             current: np.ndarray, soc: np.ndarray,
             **kwargs):
    assert isinstance(current, np.ndarray)
    assert isinstance(soc, np.ndarray)
    assert "data" in kwargs.keys()
    for col in ["ocv","r_int","r_1","r_2","c_1","c_2"]:
        assert col in kwargs["data"].columns

    pp = PP(plot_names = ["SOC v Time", "Model Voltage v Time", "Current v Time"],
             line_names = ["Modeled Data"],
             y_lim = [[0,100],
                      [1,4],
                      [int(min(current)-1), int(max(current))+1]],
             # x_lim = [0,len(current)],
             x_label = "Test Time (sec)",
             x_iterator = False)

    model_v = pd.Series(name="Model-V",dtype="float64")
    u_rc = np.zeros((2,))

    _, ocv = generate_ocv_curve(kwargs["data"]["ocv"].values)

    for i in range(len(current)):
        if (soc[i] >= 99.8 and current[i] > 0.0) or \
           (soc[i] <= 5.3 and current[i] < 0.0):

            slice1 = int(min(len(current), i + 2000))
            slice2 = int(min(len(current), i + 12000))
            mask1 = [0.0] * (slice1 - i)
            current[i: slice1] = mask1
            mask2 = current[slice1: slice2].copy() * -1.0
            current[slice1: slice2] = mask2

            delta_cap = current / 3600 * delta_t
            soc = 100 * (capacity + delta_cap.cumsum()) / capacity


        use_soc_ocv = round(soc[i], 0)
        use_soc = round(soc[i], -1)

        # for the params dataframe rounds to nearest ten
        model_v.loc[i], u_rc = model_2rc(current[i],
                                         delta_t,
                                         u_rc,
                                         ocv[int(100 - use_soc_ocv)],
                                         kwargs["data"].loc[use_soc,
                                                            "r_int"],
                                         kwargs["data"].loc[use_soc,
                                                            ["r_1", "r_2"]].values,
                                         kwargs["data"].loc[use_soc,
                                                            ["c_1", "c_2"]].values
                                         )
        pp.update(delta_t * i,
                  [ [soc[i]],
                    [model_v.loc[i]],
                    [current[i]] ])

    pp.finalize()

    return pd.DataFrame(data={"current": current,
                              "voltage": model_v,
                              "soc": soc})


def simulate(capacity, current, delta_t=1.0, **kwargs):
    '''
    float[, float], **kwargs -> pd.DataFrame, .csv file

    Simulates the li-ion cell under different current profiles
    outputs a pd.DataFrame with the resulting data from the simulation

    Parameters:
    `capacity` float
        the capacity in Ampere-hours of the cell
    `current` np.ndarray
        current profile to be used
        the array should only be around 20 values long
    `delta_t` float
        the time between data points (this is important for the ECM model)
        the value of the `delta_t` will be static
    `kwargs`
     Note if `data` in kwargs: there should be no other kwargs passed
          if `r_int`, etc in kwargs: there should be no data argument passed
        `r_int` float
            the internal resistance of the lithium-ion cell
        `r_1` and `r_2` float and float
            the resistances of the 1st and 2nd order RC pairs respectively in Ohms
        `c_1` and `c_2` float and float
            the capacitances of the 1st and 2nd order RC pairs respectively
        `ocv` list of floats
            the ocv values at 100, 90, 80, ..., 20, 10% SOC
            list should contain ten floating point values

        `data` pd.DataFrame
            DataFrame containing ECM paramaters at every 10% SOC
            I have a function in my ecm_battery_fit repository that makes this for me
            !!!!!
            The datapoints should go from 100 to 10 SOC, this function assumes
            the data is in decreasing SOC order
            !!!!!
    '''
    assert (isinstance(capacity, float) and isinstance(delta_t, float))
    assert (capacity > 1.0 and delta_t > 0.0)
    assert(len(current) > 15)

    current[3], current[6], current[15]= 0.00, 0.00, 0.00
    if current[0] >= 0.0:
        current[0] *= -1
    if current[0] >= -2.0:
        current[0] *= 2
    current_list= [0.0]
    for i in range(len(current)):
        current_list.extend([current[i]] * int(6000 // (i+1) ** 0.4))

    df_sim= pd.DataFrame(columns={"current", "voltage", "soc"})
    df_sim["current"]= current_list

    # generate soc ahead of time
    delta_cap= df_sim["current"] / 3600 * delta_t
    df_sim["soc"]= 100 * (capacity + delta_cap.cumsum()) / capacity

    if "data" in kwargs.keys():
        df_sim= lfp_cell(capacity,
                         delta_t,
                         df_sim["current"].values,
                         df_sim["soc"].values,
                         data=kwargs["data"])
    else:
        for i in ["r_int","r_1", "r_2","c_1", "c_2"]:
            assert i in kwargs.keys()
            
        df_sim= lfp_cell(capacity,
                      delta_t,
                      df_sim["current"].values,
                      df_sim["soc"].values,
                      ocv=kwargs["ocv"],
                      r_int=kwargs["r_int"],
                      r_1=kwargs["r_1"],
                      r_2=kwargs["r_2"],
                      c_1=kwargs["c_1"],
                      c_2=kwargs["c_2"])
     
    df_sim["time"] = [t * delta_t for t in range(len(df_sim))]

    return df_sim
