These are the scripts to reproduce figures in the paper "How the dynamic interplay of cortico-basal ganglia-thalamic pathways shapes the time course of deliberation and commitment" https://doi.org/10.1101/2025.03.17.643668

The CBGT network codebase (CBGTPy) utilized in this study is publicly available at https://github.com/CoAxLab/CBGTPy. Detailed installation instructions and a comprehensive list of implemented functions can be found in the README.txt file within the repository. Using CBGTPy, the firing rate data for 300 networks is generated (see folder "Data").

The DDM fit requires hssm toolbox to be installed (see https://lnccbrown.github.io/HSSM/) and see generate_ddm.ipynb in the folder "DDM" for execution in this study. 

The demo notebook data_analysis.ipynb provides instructions to process firing rate and DDM data and reproduce figures.
