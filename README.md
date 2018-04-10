# GS260_resources
Tutorials and resources for GS 260 Uncertainty Quantification in Subsurface Systems

## Installation instructions for MODFLOW tutorial

The MODFLOW and FloPY tutorial requires installation of [MODFLOW-2005](https://water.usgs.gov/ogw/modflow/mf2005.html), [Python 2.7 with scientific packages](https://www.anaconda.com/download/), and [FloPy](https://github.com/modflowpy/flopy)

### Windows:

1. Download the MODFLOW-2005 compiled executable and add its location to your environment variables path. [MODFLOW Download](https://water.usgs.gov/ogw/modflow/mf2005.html)

2. Install Python 2.7 with scientific packages (numpy, scipy, pandas, matplotlib). [Anaconda Python distribution](https://www.anaconda.com/download/)

3. Install FloPy [FloPy](https://github.com/modflowpy/flopy)
    - In a command window, enter: *pip install flopy*  

### Mac:

1. Download MODFLOW-2005 unix source code [MODFLOW Download](https://water.usgs.gov/ogw/modflow/mf2005.html)

2. Unzip the file, access the subfolder 'MF2005.1_12u/make' via the terminal. Enter the command *make*

3. Install scientific packages (numpy, pandas, matplotlib). Via the terminal, enter the following commands:
    - pip install numpy
    - pip install pandas
    - pip install matplotlib  

4. Install FloPy [FloPy](https://github.com/modflowpy/flopy)
    - pip install flopy

