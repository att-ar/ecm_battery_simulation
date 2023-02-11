import streamlit as st

import numpy as np
from numpy.polynomial import Polynomial as P
import pandas as pd

import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid

device = torch.device("cpu")

@st.cache_data
def generate_ocv_curve(ocv: list):
    '''
    numpy.polynomial.Polynomial.fit
    generates soc -> ocv mapping polynomial of degree 8
    and outputs ocv values at all unit soc values from 100% to 10%
    '''
    # assert isinstance(ocv, list)
    soc = list(range(100,-1,-10))
    assert len(ocv) == len(soc)
    assert len(ocv) == 11
    curve = P.fit(soc, ocv, 8, domain = [0,100])
    return curve.linspace(101, domain = (100, 0))

# Second Order RC-Model
def model_2rc(current, delta_t, u_rc, ocv, r_int, r, c):
    # returns the new voltage and polarization voltage
    tau_i = r * c
    u_rc = np.exp(-delta_t / tau_i) * u_rc + r * \
        (1 - np.exp(-delta_t / tau_i)) * (-current)

    return ocv + r_int * current - u_rc.sum(), u_rc


def lfp_cell(capacity: float, delta_t: float,
             current: np.ndarray, soc: np.ndarray,
             progress,
             **kwargs):
    assert isinstance(current, np.ndarray)
    assert isinstance(soc, np.ndarray)
    model_v = pd.Series(name="Model-V",dtype="float64")
    u_rc = np.zeros((2,))

    r = np.array([kwargs["r_1"], kwargs["r_2"]])
    c = np.array([kwargs["c_1"], kwargs["c_2"]])

    _, ocv = generate_ocv_curve(kwargs["ocv"])

    for i in range(len(current)):
        progress.progress((i + 1) / len(current))
        if (soc[i] >= 99.9 and current[i] > 0.0) or \
           (soc[i] <= 0.3 and current[i] < 0.0):

            slice1 = min(len(current), i + 750)
            slice2 = min(len(current), i + 3600)
            mask1 = [0.0] * (slice1 - i)
            current[i: slice1] = mask1
            mask2 = current[slice1: slice2].copy() * -1.0
            current[slice1: slice2] = mask2

            delta_cap = current / 3600 * delta_t
            soc = 100 * (capacity + delta_cap.cumsum()) / capacity

        use_soc_ocv = round(soc[i], 0)

        model_v.loc[i], u_rc = model_2rc(current[i],
                                         delta_t,
                                         u_rc,
                                         ocv[int(100 - use_soc_ocv)],
                                         kwargs["r_int"],
                                         r,
                                         c)

    return pd.DataFrame(data={"current": current,
                              "voltage": model_v,
                              "soc": soc})

def simulate(capacity, current, progress, delta_t=1.0, **kwargs):
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
    `progress` st.progress() object
        Just a progress bar from the streamlit API
    `delta_t` float
        the time between data points (this is important for the ECM model)
        the value of the `delta_t` will be static
    `kwargs`
        `r_int` float
            the internal resistance of the lithium-ion cell
        `r_1` and `r_2` float and float
            the resistances of the 1st and 2nd order RC pairs respectively in Ohms
        `c_1` and `c_2` float and float
            the capacitances of the 1st and 2nd order RC pairs respectively
        `ocv` list of floats
            the ocv values at 100, 90, 80, ..., 20, 10% SOC
            list should contain ten floating point values
    '''
    assert (isinstance(capacity, float) and isinstance(delta_t, float))
    assert (capacity > 1.0 and delta_t > 0.0)
    for i in ["r_int","r_1", "r_2","c_1", "c_2"]:
        assert i in kwargs.keys()
    assert(len(current) > 15)

    current[3], current[6], current[15]= 0.00, 0.00, 0.00
    if current[0] < 0.0:
        current[0] *= -1.0
    if current[0] < 4.0:
        current[0] *= 5.0
    current_list= [0.0] + [-capacity] * 3600
    #ensures a sweep from 100 SOC to 0 SOC, which is industry norm,
    #and required for my model to function well
    for i in range(len(current)):
        current_list.extend([current[i]] * int(3000 // (i+1) ** 0.4))

    df_sim= pd.DataFrame(columns=["current", "voltage", "soc"])
    df_sim["current"]= current_list

    # generate soc ahead of time
    delta_cap= df_sim["current"] / 3600 * delta_t
    df_sim["soc"]= 100 * (capacity + delta_cap.cumsum()) / capacity
    #sim
    df_sim= lfp_cell(capacity,
                      delta_t,
                      df_sim["current"].values,
                      df_sim["soc"].values,
                      progress,
                      ocv=kwargs["ocv"],
                      r_int=kwargs["r_int"],
                      r_1=kwargs["r_1"],
                      r_2=kwargs["r_2"],
                      c_1=kwargs["c_1"],
                      c_2=kwargs["c_2"])
    df_sim["time"] = [round(t * delta_t,1) for t in range(len(df_sim))]

    return df_sim

class LSTMNetwork(nn.Module):
        def __init__(self):
            super(LSTMNetwork, self).__init__()
            self.lstm = nn.LSTM(3, 256, 1, batch_first = True)
            self.linear_stack = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, momentum = 0.92),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, momentum = 0.92),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
                Sigmoid()
        )
        def forward(self, x):
            #lstm
            x_out, (h_n_lstm, c_n)  = self.lstm(x)
            out = self.linear_stack(h_n_lstm.squeeze())
            return out

@st.cache_resource
def load_model():
    model = LSTMNetwork().to(device)
    model.load_state_dict(torch.load("crate_model_state_dict.pth", map_location = device))
    model.eval()
    return model

