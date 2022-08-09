# Lithium-Ion Cell Simulation

Link to [Streamlit App](https://att-ar-ecm-battery-simulation-lfp-simulate-gawm41.streamlitapp.com/):

Uses a 2nd Order RC Model of the ECM in order to simulate voltage and soc behaviour of a lithium-ion cell<br>
subjected to different current profiles.

I have implemented a streamlit app in order to facilitate user input. <br>
But I will also add a Jupyter notebook with a rolling Jupyter plot. <br>

## ECM <a id = "ecm"></a>

$$ U_{1,k+1} = exp(-\Delta t/\tau_1)\cdot U_{1,k} + R_1[1 - exp(-\Delta t/\tau_1)]\cdot I_k $$

$$ U_{2,k+1} = exp(-\Delta t/\tau_2)\cdot U_{2,k} + R_2[1 - exp(-\Delta t/\tau_2)]\cdot I_k $$

$$ \tau_1 = R_1C_1 $$

$$ \tau_2 = R_2C_2 $$
 
$$ V_k = OCV - R_0I_k - \sum_{i=1}^{2}U_{i,k} $$
