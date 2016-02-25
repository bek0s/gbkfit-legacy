
import os
import random
import shutil

import astropy.io.fits as fits
import numpy


def add_gaussian_noise(data, sigma):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            data[y][x] += random.gauss(0, sigma)


def create_datasets(name, flxmap, velmap, sigmap, error_flx, error_vel, error_sig):
    
    filename_flxmap_d = "{}/{}_flxmap_d.fits".format(name, name)
    filename_flxmap_e = "{}/{}_flxmap_e.fits".format(name, name)
    filename_flxmap_m = "{}/{}_flxmap_m.fits".format(name, name)
    
    filename_velmap_d = "{}/{}_velmap_d.fits".format(name, name)
    filename_velmap_e = "{}/{}_velmap_e.fits".format(name, name)
    filename_velmap_m = "{}/{}_velmap_m.fits".format(name, name)

    filename_sigmap_d = "{}/{}_sigmap_d.fits".format(name, name)
    filename_sigmap_e = "{}/{}_sigmap_e.fits".format(name, name)
    filename_sigmap_m = "{}/{}_sigmap_m.fits".format(name, name)
    
    # Delete output folder if already exists
    if os.path.exists(name):
        shutil.rmtree(name)

    # Create output folder
    os.makedirs(name)

    # Create data maps (just a copy)
    flxmap_d = numpy.array(flxmap)
    velmap_d = numpy.array(velmap)
    sigmap_d = numpy.array(sigmap)
    
    # Add noise to data maps
    add_gaussian_noise(flxmap_d, error_flx)
    add_gaussian_noise(velmap_d, error_vel)
    add_gaussian_noise(sigmap_d, error_sig)
    
    # Create error maps
    flxmap_e = numpy.full_like(flxmap, error_flx)
    velmap_e = numpy.full_like(velmap, error_vel)
    sigmap_e = numpy.full_like(sigmap, error_sig)
    
    # Create mask maps
    flxmap_m = numpy.ones_like(flxmap)
    velmap_m = numpy.ones_like(velmap)
    sigmap_m = numpy.ones_like(sigmap)
    
    # Save flx map
    fits.writeto(filename_flxmap_d, flxmap_d, clobber=True)
    fits.writeto(filename_flxmap_e, flxmap_e, clobber=True)
    fits.writeto(filename_flxmap_m, flxmap_m, clobber=True)
    # Save vel map
    fits.writeto(filename_velmap_d, velmap_d, clobber=True)
    fits.writeto(filename_velmap_e, velmap_e, clobber=True)
    fits.writeto(filename_velmap_m, velmap_m, clobber=True)
    # Save sig map
    fits.writeto(filename_sigmap_d, sigmap_d, clobber=True)
    fits.writeto(filename_sigmap_e, sigmap_e, clobber=True)
    fits.writeto(filename_sigmap_m, sigmap_m, clobber=True)


def main():
    
    names = ["mock_1", "mock_2", "mock_3"]
    noise_flx = 0.001
    noise_vel = 2
    noise_sig = 3
    
    for name in names:
    
        filename_flxmap = "clean/{}_flxmap_mdl.fits".format(name)
        filename_velmap = "clean/{}_velmap_mdl.fits".format(name)
        filename_sigmap = "clean/{}_sigmap_mdl.fits".format(name)
        
        data_flxmap = fits.getdata(filename_flxmap)
        data_velmap = fits.getdata(filename_velmap)
        data_sigmap = fits.getdata(filename_sigmap)
        
        create_datasets(name, data_flxmap, data_velmap, data_sigmap, noise_flx, noise_vel, noise_sig)


if __name__ == "__main__":
    main()
