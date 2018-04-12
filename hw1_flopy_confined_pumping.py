# hw1_flopy_confined_pumping.py
# Author: Noah Athens
# Created: April 11, 2018
# Groundwater Model for GS 260 Quantifying Uncertainty in Subsurface Systems
# Mac or Linux users will need to change the path specification (line 53)

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from numpy.random import uniform as rand
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import sys

def add_shale_border(hcon, sand_width, cell_size, shale_hcon):
    nrows_sand = int(sand_width / cell_size)
    if nrows_sand > hcon.shape[0] - 1: nrows_sand = hcon.shape[0]
    nrows_shale = int((hcon.shape[0] - nrows_sand) / 2)
    hcon[:nrows_shale, :] = shale_hcon
    hcon[nrows_sand + nrows_shale:, :] = shale_hcon
    return hcon

def simulFFT(nx, ny, nz, mu, sill, m, lx , ly, lz):
    """ Performs unconditional simulation with specified mean, variance,
    and correlation length. Note to students: this is not the inverted covariance method
    that Jef mentioned in class. This is a more efficient implementation at the expense of
    being less clear.
    """
    if nz == 0: nz = 1 # 2D case
    xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel())).T
    centroid = points.mean(axis=0)
    length = np.array([lx, ly, lz])
    h = np.linalg.norm((points - centroid) / length, axis = 1).reshape((ny, nx, nz))

    if m == 'Exponential':
        c = np.exp(-3*h) * sill
    elif m == 'Gaussian':
        c = np.exp(-3*(h)**2) * sill

    grid = fftn(fftshift(c)) / (nx * ny * nz)
    grid = np.abs(grid)
    grid[0, 0, 0] = 0 # reference level
    ran = np.sqrt(grid) * np.exp(1j * np.angle(fftn(rand(size=(ny, nx, nz)))))
    grid = np.real(ifftn(ran * nx * ny * nz))
    std = np.std(grid)
    if nx == 1 or ny == 1 or nz == 1: grid = np.squeeze(grid)
    return grid / std * np.sqrt(sill) + mu

def run_modflow(fname, grid_dim, cell_size, hcon, ss, dhdl, pumping_rate):
    modelname = fname
    mf = flopy.modflow.Modflow(modelname, exe_name='mf2005') # Windows users
    #mf = flopy.modflow.Modflow(modelname,namefile_ext='nam',version='mf2005', exe_name='/Users/*/*/*/*/mf2005') # Mac users

    # Define the model grid
    nlay = 1
    nrow = grid_dim[0]
    ncol = grid_dim[1]
    lx = cell_size + 0. # cell size in x (meters)
    ly = cell_size + 0. # cell size in y
    ztop = 500.
    zbot = 0. # depth of the model in z
    delr = lx / ncol
    delc = ly / nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)

    # Time step parameters
    nper = 2 # number of stress periods
    perlen = [1, 7] # stress period lengths
    nstp = [1, 100] # time steps per period
    steady = [True, False]

    # Define initial conditions and boundary conditions
    initial_head = 500.
    strt = initial_head * np.ones((nlay, nrow, ncol), dtype=np.float32) # starting heads = 100

    # Conductance into and out of the model (at the boundaries)
    headleft = initial_head + 5 # Impose hydraulic gradient over the model
    headleft = initial_head + ncol * lx * dhdl / 2.0
    headright = initial_head - ncol * lx * dhdl / 2.0
    boundary = []

    if np.isscalar(hcon): hcon = hcon * np.ones(ibound.shape) # convert constant values to array
    if np.isscalar(ss): ss = ss * np.ones(ibound.shape)

    for il in range(nlay):
        for ir in range(nrow):
            condleft = hcon[il, ir, 0] * (headleft - zbot) * delc
            condright = hcon[il, ir, ncol - 1] * (headright - zbot) * delc
            boundary.append([il, ir, 0, headleft, condleft])
            boundary.append([il, ir, ncol - 1, headright, condright])

    ghb_stress_period_data = {0: boundary, 1: boundary}

    # Define source terms - e.g. a pumping well
    wel_sp1 = [[0, 25, 25, 0.]] # Well is located at cell (0, 25, 25). No pumping during the steadystate solutions
    wel_sp2 = [[0, 25, 25, pumping_rate]]
    wel_stress_period_data = {0:wel_sp1, 1:wel_sp2}
    oc_stress_period_data = {} # Output control
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            oc_stress_period_data[(kper, kstp)] = ['save head', 'save drawdown', 'save budget']

    # Define flopy objects
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                       top=ztop, botm=botm[1:],nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hcon, vka=1, ss = ss, laytyp=0, layvka=1, ipakcb=53)
    pcg = flopy.modflow.ModflowPcg(mf)
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_stress_period_data)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_stress_period_data)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=oc_stress_period_data, compact=True)

    # Write input files and run
    mf.write_input()
    success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    # Read output files
    headobj = bf.HeadFile(modelname + '.hds')
    times = headobj.get_times()
    before_pumping = np.squeeze(headobj.get_data(totim = times[0])) # squeeze to remove z dimension in 2D model
    after_pumping = np.squeeze(headobj.get_data(totim = times[-1]))
    time_series_at_well = headobj.get_ts((0, 25, 25))
    return before_pumping, after_pumping, time_series_at_well

def main():
    # Define parameters
    fname = 'mysim'
    grid_dim = [100, 100] # row, col
    cell_size = 100. # square cell size [m]
    variogram_range = 25 # [cells]
    variogram_model = 'Gaussian' # Gaussian/Exponential
    log10_mean_hcon = -1 # hydraulic conductivity mean [log10]
    log10_var_hcon = 0.5 # hydraulic conductivity variance [log10]
    hydraulic_gradient = 1e-3 # dh/ dl [m / m]
    pumping_rate = -1000 # production is negative, injection is positive [m3/d]
    sand_width = grid_dim[0] * cell_size # [m]
    specific_storage = 1e-3 # [1/m]

    nrealizations = 10
    hcon_reals = []
    bp_reals = []
    ap_reals = []
    well_reals = []
    for i in range(nrealizations):
        # Generate hydraulic conductivity spatial model
        log10_hcon = simulFFT(grid_dim[1], grid_dim[0], 1, log10_mean_hcon, log10_var_hcon, variogram_model, variogram_range, variogram_range, 1)
        hcon = 10**log10_hcon

        # Add shale border to hydraulic conductivity model
        hcon = add_shale_border(hcon, sand_width, cell_size, 1e-7)
        hcon = hcon[np.newaxis, :, :] # new dimension for modflow (layer, row, col)

        # Run groundwater model
        head_bp, head_ap, ts_well = run_modflow(fname, grid_dim, cell_size, hcon, specific_storage, hydraulic_gradient, pumping_rate)
        hcon_reals.append(hcon.flatten())
        bp_reals.append(head_bp.flatten())
        ap_reals.append(head_ap.flatten())
        well_reals.append(ts_well.flatten())

    # Save outputs to file
    hcon_reals = np.asarray(hcon_reals).T
    bp_reals = np.asarray(bp_reals).T
    ap_reals = np.asarray(ap_reals).T
    well_reals = np.asarray(well_reals).T
    print hcon_reals.shape

    np.savetxt(fname + '_hcon_model.csv', hcon_reals, delimiter=',')
    np.savetxt(fname + '_head_before_pumping.csv', bp_reals, delimiter=',')
    np.savetxt(fname + '_head_after_pumping.csv', ap_reals, delimiter=',')
    np.savetxt(fname + '_well_timeseries.csv', well_reals, delimiter=',')


if __name__ == '__main__':
    main()
