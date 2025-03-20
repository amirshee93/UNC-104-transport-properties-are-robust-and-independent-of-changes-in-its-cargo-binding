To generate the **figures**, navigate to the scripts/ folder and run the corresponding Python scripts. For example:

python3 plot_fig2.py,
python3 plot_fig3.py, ...


The **plot_fig2** script reads raw experimental data for motor protein intensities, processes and normalizes the data to account for variations in axonal volume, and produces a publication-ready figure with three panels that illustrate:

The raw UNC-104 intensity profile,
The mScarlet fluorophore intensity profile,
The normalized ratio indicating the motor protein density per unit volume.

The **plot_fig3** script reads normalized intensity data for motor protein distributions under different experimental conditions and compares them with a theoretical prediction. It illustrates:

The normalized steady-state distribution of bound motor proteins for wild-type (WT), UBA1 knockdown, and FBXB-65 knockdown conditions.
A theoretical curve defined by f(x)=xexp(−x), allowing for direct comparison between experiment and theory.
Clear labeling and axis formatting to enhance interpretability and publication quality.
