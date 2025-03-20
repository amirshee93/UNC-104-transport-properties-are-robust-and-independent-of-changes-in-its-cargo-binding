To generate the **figures**, download data, scripts and run the corresponding Python scripts. For example:

python3 plot_fig2.py,
python3 plot_fig3.py, ...

===============================================================================

The **plot_fig2** script reads raw experimental data for motor protein intensities, processes and normalizes the data to account for variations in axonal volume, and produces a publication-ready figure with three panels that illustrate:

The raw UNC-104 intensity profile, The mScarlet fluorophore intensity profile, The normalized ratio indicating the motor protein density per unit volume.

-------------------------------------------------------------------------------------------------------------------

The **plot_fig3** script reads normalized intensity data for motor protein distributions under different experimental conditions and compares them with a theoretical prediction. It illustrates:

The normalized steady-state distribution of bound motor proteins for wild-type (WT), UBA1 knockdown, and FBXB-65 knockdown conditions. A theoretical curve defined by f(x)=x*exp(−x), allowing for direct comparison between experiment and theory.

-------------------------------------------------------------------------------------------------------------------

The **plot_fig4** script reads raw FRAP recovery data and effective diffusivity measurements from multiple experiments and compares them with theoretical predictions of fluorescence recovery dynamics. It illustrates:

The spatial recovery profiles of fluorescence intensity at multiple time points (1 s, 31 s, and 92 s) with overlaid theoretical curves. The temporal evolution of fluorescence recovery at the FRAP center (x=0), with characteristic recovery times t_{1/4} and t_{1/2} clearly indicated. A violin plot comparing effective diffusivities for wild-type (WT), UBA1 knockdown, and FBXB-65 knockdown conditions, allowing for direct comparison between experimental results and theory.
