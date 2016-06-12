
import os
import random
import shutil

import astropy.io.fits as fits
import numpy


def add_gaussian_noise(data, sigma):
    if len(data.shape) == 2:
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                data[y][x] += random.gauss(0, sigma)
    if len(data.shape) == 3:
        for z in range(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    data[z][y][x] += random.gauss(0, sigma)


def create_datasets(name, flxcube, flxmap, velmap, sigmap, flxcube_error, flxmap_error, velmap_error, sigmap_error):
    
    filename_flxcube_d = "{}/{}_flxcube_d.fits".format(name, name)
    filename_flxcube_e = "{}/{}_flxcube_e.fits".format(name, name)
    filename_flxcube_m = "{}/{}_flxcube_m.fits".format(name, name)
    
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
    flxcube_d = numpy.array(flxcube)
    flxmap_d = numpy.array(flxmap)
    velmap_d = numpy.array(velmap)
    sigmap_d = numpy.array(sigmap)
    
    # Add noise to data maps
    add_gaussian_noise(flxcube_d, flxcube_error)
    add_gaussian_noise(flxmap_d, flxmap_error)
    add_gaussian_noise(velmap_d, velmap_error)
    add_gaussian_noise(sigmap_d, sigmap_error)
    
    # Create error maps
    flxcube_e = numpy.full_like(flxcube, flxcube_error)
    flxmap_e = numpy.full_like(flxmap, flxmap_error)
    velmap_e = numpy.full_like(velmap, velmap_error)
    sigmap_e = numpy.full_like(sigmap, sigmap_error)
    
    # Create mask maps
    flxcube_m = numpy.ones_like(flxcube)
    flxmap_m = numpy.ones_like(flxmap)
    velmap_m = numpy.ones_like(velmap)
    sigmap_m = numpy.ones_like(sigmap)
    
    # Save flx cube
    fits.writeto(filename_flxcube_d, flxcube_d, clobber=True)
    fits.writeto(filename_flxcube_e, flxcube_e, clobber=True)
    fits.writeto(filename_flxcube_m, flxcube_m, clobber=True)
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

    flxcube_noise = 0.002
    flxmap_noise = 0.01
    velmap_noise = 2
    sigmap_noise = 3
    
    for name in names:
    
        filename_flxcube = "clean/{}_flxcube_clean.fits".format(name)
        filename_flxmap = "clean/{}_flxmap_clean.fits".format(name)
        filename_velmap = "clean/{}_velmap_clean.fits".format(name)
        filename_sigmap = "clean/{}_sigmap_clean.fits".format(name)
        
        flxcube = fits.getdata(filename_flxcube)
        flxmap = fits.getdata(filename_flxmap)
        velmap = fits.getdata(filename_velmap)
        sigmap = fits.getdata(filename_sigmap)
        
        create_datasets(name, 
                        flxcube, 
                        flxmap, 
                        velmap, 
                        sigmap, 
                        flxcube_noise, 
                        flxmap_noise, 
                        velmap_noise, 
                        sigmap_noise)


if __name__ == "__main__":
    main()
