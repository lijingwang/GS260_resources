# check_modflow_installed.py
# Created: April 5th, 2018 for GS 260 tutorial
# Based on FloPy tutorial: https://modflowpy.github.io/flopydoc/tutorial1.html

import sys
import numpy as np
import flopy

def check_modflow():
    # Assign name and create modflow model object
    modelname = 'tutorial1'
    mf = flopy.modflow.Modflow(modelname, exe_name='mf2005') # Windows users
    #mf = flopy.modflow.Modflow(modelname,namefile_ext='nam',version='mf2005', exe_name='/Users/*/*/mf2005') # MAC users

    # Model domain and grid definition
    Lx = 1000.
    Ly = 1000.
    ztop = 0.
    zbot = -50.
    nlay = 1
    nrow = 10
    ncol = 10
    delr = Lx/ncol
    delc = Ly/nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)

    # Create the discretization object
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:])

    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.
    strt[:, :, -1] = 0.
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(mf, hk=10., vka=10., ipakcb=53)

    # Add OC package to the MODFLOW model
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)

    # Write the MODFLOW model input files
    mf.write_input()

    # Run the MODFLOW model
    success, buff = mf.run_model()

    if not success:
        raise Exception('MODFLOW did not terminate normally.')
    else:
        return True

def main():
    print 'Testing modflow installed...'
    if(check_modflow()):
        print '\n\nMODFLOW and other packages are installed correctly :)'

if __name__=='__main__':
    main()
