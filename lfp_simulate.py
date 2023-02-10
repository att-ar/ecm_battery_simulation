import streamlit as st
import plotly.express as px

from datetime import datetime

import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid

from rolling_and_plot import normalize, rolling_split, validate
from ecm_model import *

device = torch.device("cpu")

st.title("Lithium-Ion Cell Simulator Using the 2nd Order Equivalent Circuit Model")
st.markdown(
    "The purpose of this simulator is to generate SOC, Voltage, and Current data for lithium-ion cells.")
st.markdown("Open Sidebar for Static ECM Parameterization.")

tab = st.tabs(["Simulation","Graphs","LSTM Predictions"])

sidebar = st.sidebar
with sidebar:  # the sidebar of the GUI
    r_int = st.number_input(label="Internal Resistance mΩ", value = 1.3) / 1000
    # RC-pairs for 2nd order ECM
    rc = st.columns(2)
    with rc[0]:
        r_1 = st.number_input(
            label="Resistance of R1 mΩ", value=1.20) / 1000
        c_1 = st.number_input(
            label="Capacitance of C1 F", value=12000)
    with rc[1]:
        r_2 = st.number_input(
            label="Resistance of R2 mΩ", value=1.0) / 1000
        c_2 = st.number_input(
            label="Capacitance of C2 F", value=120000)

    if not r_2:
        assert r_1 != 0, "Change 1st RC pair first"
    if not c_2:
        assert c_1 != 0, "Change 1st RC pair first"
    capacity = st.number_input(label="Battery Capacity inAh", value=20.0)

    ocv = st.text_input(
        label="11 OCV values from 100 to 0 SOC split by commas:")
    "Example OCV: 3.557, 3.394, ..., 3.222, 3.174, 2.750"

    cur = st.columns(2)
    with cur[0]:
        min_I = st.number_input(
                    label = "Most (-) current (A)",
                    value = -10.0, max_value = 0.0)
    with cur[1]:
        max_I = st.number_input(
                    label = "Most (+) current (A)",
                    value = 10.0, min_value = 0.0)

    start = st.checkbox(label="Run Simulation")

#------------------------------
#converting dataframes to csv/parquet
@st.cache
def convert_df(df, format = "csv"):
    if format == "parquet":
        return df.to_parquet()
    elif format == "csv":
        return df.to_csv().encode('utf-8')
#------------------------------

if start:
    if len(ocv) > 0:
        ocv = ocv.split(",")
        assert len(ocv) == 11, "Need 11 OCV values"
        for i in ocv:
            i = i.strip()
            assert i.replace(".","").isnumeric(), "Need numerical values"
        ocv = [float(val) for val in ocv]
    assert abs(min_I) < capacity / 1.2, "Current cannot exceed C/1.2"
    assert abs(max_I) < capacity / 1.2, "Current cannot exceed C/1.2"
    current = np.array((max_I - min_I) * np.random.random_sample(24) + min_I).round(2)

    with tab[0]:
        "Progress Bar:"
        progress = st.progress(0)

    #sim
    try:
        del df_sim
    except NameError:
        df_sim = simulate(capacity, current, progress, ocv = ocv, r_int = r_int,
                          r_1 = r_1, c_1 = c_1, r_2 = r_2, c_2 = c_2
                         )

    

    with tab[0]:
        "Check the other Tabs for more!"
        st.dataframe(data = df_sim)


    with sidebar:
        file_date = datetime.today().strftime("%Y_%m_%d")
        file_name = st.text_input(label = "File name for download", value = f"simulated_data_{file_date}")
        st.write("Download Data:")
        download_columns = st.columns(2)
        with download_columns[0]:
            st.download_button(
                "CSV",
                convert_df(df_sim, format = "csv"),
                file_name,
                key='csv_data'
            )
        with download_columns[1]:
            st.download_button(
                "Parquet",
                convert_df(df_sim, format = "parquet"),
                file_name,
                key='parquet_data'
            )
        " "
        " "
    one = px.line(data_frame = df_sim, x= "time", y = "soc", title = "SOC v Time")
    one["data"][0]["line"]["color"] = "red"
    two = px.line(data_frame = df_sim, x= "time", y = "voltage", title = "Voltage v Time")
    two["data"][0]["line"]["color"] = "pink"
    three = px.line(data_frame = df_sim, x = "time", y = "current", title = "Current v Time")
    three["data"][0]["line"]["color"] = "lightgreen"

    with tab[1]:
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

    with tab[2]:
        with st.container():
            "Prediction Progress"
            prediction_bar = st.progress(0)

        lstm_cols = st.columns(2)

        df_sim_norm = normalize(df_sim, capacity)
        x_set, y_set = rolling_split(df_sim_norm)
        set_dataloader = [set for set in zip(x_set,y_set)]

        model = LSTMNetwork().to(device)
        model.load_state_dict(torch.load("crate_model_state_dict.pth", map_location = device))

        model.eval()
        with lstm_cols[0]:
            visualize, fig = validate(model, set_dataloader, prediction_bar)
            st.dataframe(visualize)
        with lstm_cols[1]:
            st.plotly_chart(fig)
