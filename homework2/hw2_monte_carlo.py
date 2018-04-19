# hw2_monte_carlo.py
# Author: Noah Athens
# Created: April 18, 2018
# Groundwater Model for GS 260 Quantifying Uncertainty in Subsurface Systems
# Mac or Linux users will need to change the path specification (line 53)

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from numpy.random import uniform as rand
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
from scipy.stats import skewnorm
import pandas as pd
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import sys
pd.options.mode.chained_assignment = None  # default='warn'

def add_shale_border(hcon, shale_width, cell_size, shale_hcon):
    nrows_shale = int(shale_width / cell_size)
    nrows_sand = hcon.shape[0] - 2 * nrows_shale
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

def transform_distribution(grid, new_distribution):
    """ Transforms grid to new distribution."""
    old_distribution = np.sort(grid.flatten())
    new_distribution = np.sort(np.random.choice(new_distribution, size=grid.size))
    d = dict(zip(old_distribution, new_distribution))
    new_grid = np.vectorize(d.get)(grid)
    return new_grid

def resistivity_forward_model(grid, hconshale):
    num_shale = grid[grid==hconshale].size
    grid = median_filter(grid, size=11) # Apply some different filters
    grid = uniform_filter(grid, size = 5)
    new_grid = gaussian_filter(grid, sigma=3, order=0)
    # Create a "resistivity" distribution
    peak1 = np.random.normal(loc=0.2, scale=0.1, size=num_shale)
    peak2 = np.random.normal(loc=1.5, scale=0.5, size=grid.size - num_shale)
    new_distribution = np.concatenate((peak1, peak2))
    new_distribution[new_distribution < 0] = 0
    new_grid = transform_distribution(new_grid, new_distribution)
    return new_grid

def run_modflow(fname, grid_dim, cell_size, hcon, ss, dhdl, pumping_rate, well_location):
    modelname = fname
    mf = flopy.modflow.Modflow(modelname, exe_name='mf2005') # Windows users
    #mf = flopy.modflow.Modflow(modelname,namefile_ext='nam',version='mf2005', exe_name='/Users/*/*/*/*/mf2005') # Mac users

    # Define the model grid
    nlay = 1
    nrow = grid_dim[0]
    ncol = grid_dim[1]
    lx = cell_size + 0. # cell size in x (meters)
    ly = cell_size + 0. # cell size in y
    ztop = 0.
    zbot = -3000. # depth of the model in z
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
    wel_sp1 = [[0, well_location[0], well_location[1], 0.]] # Well is located at cell (0, 25, 25). No pumping during the steadystate solutions
    wel_sp2 = [[0, well_location[0], well_location[1], pumping_rate]]
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
    time_series_at_well = headobj.get_ts((0, well_location[0], well_location[1]))
    return before_pumping, after_pumping, time_series_at_well


def main():
    # INPUTS
    fname_parameters = 'example_monte_carlo_parameters.csv'
    df = pd.read_csv(fname_parameters)
    fname = 'mysim' # Name your simulation
    wellLocation = [48, 30] # Use these values: [48, 30], [30, 55], [45, 57]

    # Define parameters
    grid_dim = [100, 100] # row, col
    cell_size = 100. # square cell size [m]
    variogramRange = df['variogramRange'].values / cell_size
    variogramModel = df['variogramModel'].values
    hconlogMean = df['hconlogMean'].values
    hconlogVariance = df['hconlogVariance'].values
    hydraulicGradient = df['hydraulicGradient'].values
    shaleWidth = df['shaleWidth'].values
    specificStorage = 10**df['specificStorage'].values
    pumpingRate = -1000.
    nrealizations = len(df)
    hcon_reals = []
    bp_reals = []
    ap_reals = []
    well_reals = []
    resistivity_reals = []
    df['success'] = False
    for i in range(nrealizations):
        print 'Running realization: ' + str(i + 1)
        try:
            # Generate hydraulic conductivity spatial model
            log10_hcon = simulFFT(grid_dim[1], grid_dim[0], 1, hconlogMean[i], hconlogVariance[i], variogramModel[i], variogramRange[i], variogramRange[i], 1)
            hcon = 10**log10_hcon

            # Add shale border to hydraulic conductivity model
            hconshale = 1e-7
            hcon = add_shale_border(hcon, shaleWidth[i], cell_size, hconshale)
            resistivity = resistivity_forward_model(hcon, hconshale)

            # Run groundwater model
            hcon = hcon[np.newaxis, :, :] # new dimension for modflow (layer, row, col)
            head_bp, head_ap, ts_well = run_modflow(fname, grid_dim, cell_size, hcon, specificStorage[i], hydraulicGradient[i], pumpingRate, wellLocation)
            hcon_reals.append(hcon.flatten())
            resistivity_reals.append(resistivity.flatten())
            bp_reals.append(head_bp.flatten())
            ap_reals.append(head_ap.flatten())
            well_reals.append(ts_well.flatten())
            df['success'].iloc[i] = True
        except:
            pass

    df = df.drop(df[df.success == False].index) # Remove simulations that didn't terminate normally from prior parameters
    df.drop('success', axis=1, inplace=True)

    # Save outputs to file
    hcon_reals = np.asarray(hcon_reals).T # Hydraulic Conductivity model
    resistivity_reals = np.asarray(resistivity_reals).T # Resistivity forward model
    bp_reals = np.asarray(bp_reals).T # Before pumping
    ap_reals = np.asarray(ap_reals).T # After pumping
    well_reals = np.asarray(well_reals).T # Well time-series

    np.savetxt(fname + '_hcon_model.csv', hcon_reals, delimiter=',')
    np.savetxt(fname + '_resistivity_model.csv', resistivity_reals, delimiter=',')
    np.savetxt(fname + '_head_before_pumping.csv', bp_reals, delimiter=',')
    np.savetxt(fname + '_head_after_pumping.csv', ap_reals, delimiter=',')
    np.savetxt(fname + '_well_timeseries.csv', well_reals, delimiter=',')
    df.to_csv(fname + '_prior_parameters.csv', index=False)

if __name__ == '__main__':
    main()
