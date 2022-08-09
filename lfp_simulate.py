import schemdraw
import schemdraw.elements as elm
from schemdraw.elements import Resistor as R

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("LFP Chemistry Lithium-Ion Cell Simulator")
st.markdown(
    "The purpose of this simulator is to generate SOC data for li-ion cells.")
st.markdown("Open Sidebar for Static ECM Parameterization.")

sidebar = st.sidebar
with sidebar:  # the sidebar of the GUI
    "ECM Model Parameters:"
    r_int = st.number_input(label="Internal Resistance mΩ") / 1000
    # RC-pairs for 2nd order ECM
    rc = st.columns(2)
    with rc[0]:
        r_1 = st.number_input(
            label="Resistance of R1 mΩ", value=0.0) / 1000
        c_1 = st.number_input(
            label="Capacitance of C1", value=0.0)
    with rc[1]:
        r_2 = st.number_input(
            label="Resistance of R2 mΩ", value=0.0) / 1000
        c_2 = st.number_input(
            label="Capacitance of C2", value=0.0)

    if not r_2:
        assert(r_1 != 0, "Change 1st RC pair first")
    if not c_2:
        assert(c_1 != 0, "Change 1st RC pair first")

    "Battery Characteristics"
    capacity = st.number_input(label="Battery Capacity in Ah", value=18.254)
    ocv = st.text_input(
        label="10 OCV values from 100 to 10 SOC separated by commas:")
    "Example of OCV input: 3.557, 3.459, ..., 3.222, 3.198"
    if len(ocv) > 0:
        ocv = ocv.split(",")
        assert len(ocv) == 10, "Need 10 OCV values"
        for i in ocv:
            i = i.strip()
            assert i.replace(".","").isnumeric(), "Need numerical values"
        ocv = [float(val) for val in ocv]

    start = st.checkbox(label="Check to Run Simulation")

# Second Order RC-Model
def model_2rc(current, delta_t, u_rc, ocv, r_int, r, c):
    # returns the new voltage and polarization voltage
    tau_i = r * c
    u_rc = np.exp(-delta_t / tau_i) * u_rc + r * \
        (1 - np.exp(-delta_t / tau_i)) * current

    return ocv - r_int * abs(current) - u_rc.sum(), u_rc


def lfp_cell(capacity: float, delta_t: float,
             current: pd.Series, soc: pd.Series,
             **kwargs):

    model_v = pd.Series(ocv, name="Model-V")
    u_rc = np.zeros((2,))

    if "data" in kwargs.keys():  # a dataframe with parameters was passed
        for i in current.index:
            if soc[i] >= 99.9 and current[i] > 0.0:
                current[i: i + 10] *= 0.0
                current[i + 10: i + 50] *= -1.0
                delta_cap = current / 3600 * delta_t
                soc = 100 * (capacity + delta_cap.cumsum()) / capacity
            if soc[i] <= 5.1 and current[i] < 0.0:
                current[i: i + 400] *= 0.0
                current[i + 400: min(len(current), i + 3000)] *= -1.0
                delta_cap = current / 3600 * delta_t
                soc = 100 * (capacity + delta_cap.cumsum()) / capacity

            use_soc = round(soc[i], -1)
            # for the params dataframe rounds to nearest ten
            model_v.loc[i], u_rc = model_2rc(current[i],
                                             delta_t,
                                             u_rc,
                                             kwargs["data"].loc[use_soc,
                                                                "ocv"].values,
                                             kwargs["data"].loc[use_soc,
                                                                "r_int"].values,
                                             kwargs["data"].loc[use_soc,
                                                                ["r_1", "r_2"]].values,
                                             kwargs["data"].loc[use_soc,
                                                                ["c_1", "c_2"]].values
                                             )
    else:
        r = np.array([kwargs["r_1"], kwargs["r_2"]])
        c = np.array([kwargs["c_1"], kwargs["c_2"]])
        for i in current.index:
            if soc[i] >= 99.9 and current[i] > 0.0:
                current[i: i + 400] *= 0.0
                current[i + 400: min(len(current), i + 3000)] *= -1.0
                delta_cap = current / 3600 * delta_t
                soc = 100 * (capacity + delta_cap.cumsum()) / capacity
            if soc[i] <= 5.1 and current[i] < 0.0:
                current[i: i + 400] *= 0.0
                current[i + 400: min(len(current), i + 3000)] *= -1.0
                delta_cap = current / 3600 * delta_t
                soc = 100 * (capacity + delta_cap.cumsum()) / capacity

            model_v.loc[i], u_rc = model_2rc(current[i],
                                             delta_t,
                                             u_rc,
                                             kwargs["ocv"][10 -
                                                 int(round(soc[i], -1) / 10)],
                                             kwargs["r_int"],
                                             r, c)
        # the ocv has that indexing so that the model uses the appropriate one
        # based on the SOC it is at
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
    `delta_t` float
        the time between data points (this is important for the ECM model)
        the value of the `delta_t` will be static
    `kwargs`
        Note if data in kwargs: there should be no other kwargs passed
             if r_int, etc in kwargs: there should be no data argument passed
        `r_int` float
            the internal resistance of the lithium-ion cell
        `r_1` and `r_2` float and float
            the resistances of the 1st and 2nd order RC pairs respectively in Ohms
        `c_1` and `c_2` float and float
            the capacitances of the 1st and 2nd order RC pairs respectively

        `data` pd.DataFrame
            DataFrame containing ECM paramaters at every 10% SOC
            I have a function in my ecm_battery_fit repository that makes this for me
    '''
    assert (isinstance(capacity, float) and isinstance(delta_t, float))
    assert (capacity > 16.0 and delta_t > 0.0)

    df_sim= pd.DataFrame(columns={"current", "voltage", "soc"})


    current[3], current[6], current[15]= 0.00, 0.00, 0.00
    if current[0] >= 0.0:
        current[0] *= -1
    if current[0] >= -2.0:
        current[0] *= 2

    # sim current
    current_list= [0.0]
    for i in range(len(current)):
        current_list.extend([current[i]] * int(6000 // (i+1) ** 0.4))
    df_sim["current"]= current_list

    # generate soc ahead of time
    delta_cap= df_sim["current"] / 3600 * delta_t
    df_sim["soc"]= 100 * (capacity + delta_cap.cumsum()) / capacity

    if "data" in kwargs.keys():
        df_sim= lfp_cell(capacity,
                         delta_t,
                         df_sim["current"],
                         df_sim["soc"],
                         data=kwargs["data"])
    else:
        for i in ["r_int","r_1", "r_2","c_1", "c_2","ocv"]:
            assert i in kwargs.keys()
        df_sim= lfp_cell(capacity,
                          delta_t,
                          df_sim["current"],
                          df_sim["soc"],
                          ocv=kwargs["ocv"],
                          r_int=kwargs["r_int"],
                          r_1=kwargs["r_1"],
                          r_2=kwargs["r_2"],
                          c_1=kwargs["c_1"],
                          c_2=kwargs["c_2"])
        df_sim["time"] = [t * delta_t for t in range(len(df_sim))]

    return df_sim

if start:
    current = np.array((10 + 10) * np.random.random_sample(24) - 10).round(2)
    df_sim = simulate(capacity, current, ocv = ocv, r_int = r_int,
                      r_1 = r_1, c_1 = c_1, r_2 = r_2, c_2 = c_2
                     )
    with schemdraw.Drawing() as s:
        D = {}
        lst = [[r_1,c_1],[r_2,c_2]]
        s.config(unit = 2)
        s.add( Dot1 := elm.Dot())
        s.add( V := elm.SourceV()).label("LFP Cell")
        s += elm.Dot()
        s.add( R_internal := R(label = str(round( 1000 * r_int,3)) + "mΩ"))
        s += elm.Line().right()
        s += elm.Dot()
        for i in range(len(lst)):
            s.push()
            s += elm.Line().down().dot()
            s += R().label(label = f"R{i+1} " + str(round(1000 * lst[i][0], 3)) + " mΩ",
            loc = "bottom"
            ).right().dot()
            s += elm.Line().up()
            D[i] = elm.Dot()
            s += D[i]
            s.pop()
            s.push()
            s += elm.Line().up().dot()
            s.add( elm.Capacitor(label = f"C{i+1} " + str(round(lst[i][1], 3)) + " F").right())
            s += elm.Dot()
            s += elm.Line().down().dot()
            s.pop()
            s += elm.Line().at(D[i].start).right()
        s += elm.Line().toy(V.start)
        s += elm.Dot()
        s += elm.SourceI(loc="bottom").left()
        s += elm.Line().tox(V.start)

        image = s.get_imagedata("jpg")
    st.image(image)

    fig, ax = plt.subplots(2)
    ax[0].set_ylabel("SOC (%)", fontsize = 12 )
    ax[0].set_xlabel("Time (sec)", fontsize = 12)
    ax[0].plot(df_sim["time"].values, df_sim["soc"].values, "r--")
    ax[0].set_title("SOC vs Time")
    ax[1].set_ylabel("Voltage (V)", fontsize = 12)
    ax[1].set_xlabel("Time (sec)", fontsize = 12)
    ax[1].plot(df_sim["time"].values, df_sim["voltage"].values, "b--")
    ax[1].set_title("Voltage vs Time")
    fig.tight_layout()
    st.pyplot(fig)
