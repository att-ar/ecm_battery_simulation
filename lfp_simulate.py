import schemdraw
import schemdraw.elements as elm
from schemdraw.elements import Resistor as R

import streamlit as st
# import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
from numpy.polynomial import Polynomial as P
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rolling_and_plot import normalize, rolling_split, validate

st.title("Lithium-Ion Cell Simulator Using the 2nd Order Equivalent Circuit Model")
st.markdown(
    "The purpose of this simulator is to generate SOC, Voltage, and Current data for lithium-ion cells.")
st.markdown("Open Sidebar for Static ECM Parameterization.")

tab = st.tabs(["Simulation","DataFrame","Graphs","LSTM Predictions"])

sidebar = st.sidebar
with sidebar:  # the sidebar of the GUI
    r_int = st.number_input(label="Internal Resistance mΩ", value = 2.3) / 1000
    # RC-pairs for 2nd order ECM
    rc = st.columns(2)
    with rc[0]:
        r_1 = st.number_input(
            label="Resistance of R1 mΩ", value=2.7) / 1000
        c_1 = st.number_input(
            label="Capacitance of C1 F", value=12000)
    with rc[1]:
        r_2 = st.number_input(
            label="Resistance of R2 mΩ", value=2.1) / 1000
        c_2 = st.number_input(
            label="Capacitance of C2 F", value=120000)

    if not r_2:
        assert(r_1 != 0, "Change 1st RC pair first")
    if not c_2:
        assert(c_1 != 0, "Change 1st RC pair first")

    capacity = st.number_input(label="Battery Capacity in Ah", value=18.254)

    ocv = st.text_input(
        label="10 OCV values from 100 to 10 SOC split by commas:")
    "Example OCV: 3.557, 3.394, ..., 3.222, 3.174"

    cur = st.columns(2)
    with cur[0]:
        min_I = st.number_input(
                    label = "Most (-) current (A)",
                    value = -1.0, max_value = 0.0)
    with cur[1]:
        max_I = st.number_input(
                    label = "Most (+) current (A)",
                    value = 1.0, min_value = 0.0)

    start = st.checkbox(label="Run Simulation")

def generate_ocv_curve(ocv: list):
    '''
    numpy.polynomial.Polynomial.fit
    generates soc -> ocv mapping polynomial of degree 8
    and outputs ocv values at all unit soc values from 100% to 10%
    '''
    # assert isinstance(ocv, list)
    soc = list(range(100,0,-10))
    assert len(ocv) == len(soc)
    assert len(ocv) == 10
    curve = P.fit(soc, ocv, 8, domain = [5,100])
    return curve.linspace(96, domain = (100, 5))

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
           (soc[i] <= 5.3 and current[i] < 0.0):

            slice1 = min(len(current), i + 2000)
            slice2 = min(len(current), i + 12000)
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
    if current[0] >= 0.0:
        current[0] *= -1
    if current[0] >= -2.0:
        current[0] *= 2
    current_list= [0.0]
    for i in range(len(current)):
        current_list.extend([current[i]] * int(1500 // (i+1) ** 0.4))

    df_sim= pd.DataFrame(columns={"current", "voltage", "soc"})
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
#------------------------------
#converting dataframes to csv
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
#------------------------------

if start:
    if len(ocv) > 0:
        ocv = ocv.split(",")
        assert len(ocv) == 10, "Need 10 OCV values"
        for i in ocv:
            i = i.strip()
            assert i.replace(".","").isnumeric(), "Need numerical values"
        ocv = [float(val) for val in ocv]
    assert(abs(min_I) < capacity / 1.2, "Current cannot exceed C/1.2")
    assert(abs(max_I) < capacity / 1.2, "Current cannot exceed C/1.2")
    current = np.array((max_I - min_I) * np.random.random_sample(24) + min_I).round(2)
    
    with tab[0]:
        "Progress Bar:"
        progress = st.progress(0)

        #circuit
        with schemdraw.Drawing() as s:
            D = {}
            lst = [[r_1,c_1],[r_2,c_2]]
            s.config(unit = 2)
            s.add( V := elm.SourceV()).label("LFP Cell")
            s.add( R_internal := R(label = str(round( 1000 * r_int,3)) + " mΩ"))
            s += elm.Line().right()
            s += elm.Dot()
            for i in range(len(lst)):
                s.push()
                s += elm.Line().down()
                s += R().label(label = f"R{i+1}: " + str(round(1000 * lst[i][0], 3)) + " mΩ",
                loc = "bottom"
                ).right()
                s += elm.Line().up()
                D[i] = elm.Dot()
                s += D[i]
                s.pop()
                s.push()
                s += elm.Line().up()
                s.add( elm.Capacitor(label = f"C{i+1}: " + str(round(lst[i][1], 3)) + " F").right())
                s += elm.Line().down()
                s.pop()
                s += elm.Line().at(D[i].start).right()
                if i == 0:
                    s += elm.Dot()
            s += elm.Line().toy(V.start)
            s += elm.SourceI(loc="bottom").left()
            s += elm.Line().tox(V.start)

            image = s.get_imagedata("svg")

        st.image("<svg xmlns=image></svg>", output_format = "PNG")

    #sim
    df_sim = simulate(capacity, current, progress, ocv = ocv, r_int = r_int,
                      r_1 = r_1, c_1 = c_1, r_2 = r_2, c_2 = c_2
                     )

    csv = convert_df(df_sim)
    
    with tab[1]:
        st.dataframe(data = df_sim)

    with sidebar:
        file_name = st.text_input(label = "file name for csv", value = "simulated_data.csv")
        st.download_button(
            "Download Simulated Data",
            csv,
            file_name,
            "text/csv",
            key='simulated-data'
        )
        " "
        " "
    one = px.line(data_frame = df_sim, x= "time", y = "soc", title = "SOC v Time")
    one["data"][0]["line"]["color"] = "red"
    two = px.line(data_frame = df_sim, x= "time", y = "voltage", title = "Voltage v Time")
    two["data"][0]["line"]["color"] = "pink"
    three = px.line(data_frame = df_sim, x = "time", y = "current", title = "Current v Time")
    three["data"][0]["line"]["color"] = "lightgreen"
    
    with tab[2]:
        st.plotly_chart(one)
        st.plotly_chart(two)
        st.plotly_chart(three)

    #plot
#     fig, ax = plt.subplots(3, sharex=True, figsize = (12,9))
#     ax[0].set_ylabel("SOC (%)", fontsize = 12 )
#     ax[0].set_xlabel("Time (sec)", fontsize = 12)
#     ax[0].plot(df_sim["time"].values, df_sim["soc"].values, "r-")
#     ax[0].set_title("SOC vs Time")
#     ax[0].set_ylim([0,105])
#     ax[0].set_yticks(list(range(0,105,20)))

#     ax[1].set_ylabel("Voltage (V)", fontsize = 12)
#     ax[1].set_xlabel("Time (sec)", fontsize = 12)
#     ax[1].plot(df_sim["time"].values, df_sim["voltage"].values, "b-")
#     ax[1].set_title("Voltage vs Time")
#     ax[1].set_ylim([1.9,4])
#     ax[1].set_yticks(np.arange(2.0,4.1,0.4))

#     ax[2].set_ylabel("Current (A)", fontsize = 12 )
#     ax[2].set_xlabel("Time (sec)", fontsize = 12)
#     ax[2].plot(df_sim["time"].values, df_sim["current"].values, "g-")
#     ax[2].set_title("Current vs Time")
#     ax[2].set_yticks(list(range(int(min_I),
#                                 int(max_I + 4),
#                                 4)))
#     fig.tight_layout()
#     st.pyplot(fig)

    #-----------------------------------
    # LSTM Model
    with tab[3]:
        prediction_bar = st.progress(0)
        df_sim_norm = normalize(df_sim)
        x_set, y_set = rolling_split(file)
        set_dataloader = [set for set in zip(x_set,y_set)]

        model = torch.jit.load("model_scripted.pth")
        model.eval()

        visualize, fig = validate(model, set_dataloader, prediction_bar)
        st.dataframe(visualize)
        st.plotly_chart(fig)

