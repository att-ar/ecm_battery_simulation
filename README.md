# Lithium-Ion Cell Simulation

Link to [Streamlit App](https://att-ar-ecm-battery-simulation-lfp-simulate-gawm41.streamlitapp.com/) with an integrated PyTorch LSTM model.

There is also a Jupyter Notebook that uses the self-updating ProgressPlot from JupyterPlot:<br> *simulate_cell.ipynb*

Uses a 2nd Order RC Model of the ECM in order to simulate voltage and soc behaviour of a lithium-ion cell<br>
subjected to different current profiles.

## ECM <a id = "ecm"></a>

$2^{nd}$ order:

$$ U_{1,k+1} = exp(-\Delta t/\tau_1)\cdot U_{1,k} + R_1(1 - exp(-\Delta t/\tau_1))\cdot I_k $$

$$ U_{2,k+1} = exp(-\Delta t/\tau_2)\cdot U_{2,k} + R_2(1 - exp(-\Delta t/\tau_2))\cdot I_k $$

$$ \tau_1 = R_1C_1 $$

$$ \tau_2 = R_2C_2 $$
 
$$ V_k = OCV - R_0I_k - \sum_{i=1}^{2}U_{i,k} $$

Where:

 - $R_0$ is the internal resistance of the lithium-ion cell
 - $R_i$ is the resistance of the resistor in the $i^{th}$ RC pair
 - $C_i$ is the capacitance of the capacitor in the $i^{th}$ RC pair
 - $\tau_i$ is the time constant of the $i^{th}$ resistor-capacitor (RC) pair
 - $U_{i,k}$ is the polarization voltage, at time $t = k$ of the $i^{th}$ RC pair 
 - $V_k$ is the lithium-ion cell's voltage at time $t = k$
 - $I_k$ is the current flowing into or from the cell at time $t = k$
 - $\Delta t$ is the time interval between times $t = k+1$ and $t = k$
 
