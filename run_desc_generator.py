"""Generate descriptors."""

from os import path
import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from fep_utils import (list_of_directories,
                       list_of_files,
                       geometry_to,
                       lammps_to,
                       to_poscar)

def gen_dscribe(samples, rcut, sigma):

    desc_generator = SOAP(
        species=['H', 'C'],
        rcut=rcut,
        nmax=8,
        lmax=8,
        sigma=sigma,
        crossover=True,
        periodic=True,
    )
    for sample in samples:
        print('creating soap: '+sample)
        files = list_of_files(sample)
        if 'geometry.in.next_step' in files:
            data = geometry_to(sample + '/geometry.in.next_step')
        if 'lmp.data.relax' in files:
            data = lammps_to(sample + '/lmp.data.relax')
        to_poscar(*data, sample + '/POSCAR')
        data = read(
                sample + '/POSCAR',
                format='vasp',
                )
        soap = desc_generator.create(data, n_jobs=1)
        desc = soap.data
        np.savez(
                sample + '/soap',
                desc=desc,
                )

def norm_descriptor(samples, outfile):
    desc_all = []
    for sample in samples:
        data = np.load(
                sample+'/soap.npz',
                )
        desc_each = np.reshape(data['desc'], data['desc'].shape).tolist()
        desc_all = desc_all + desc_each

    desc_all = np.array(desc_all)
    desc_mean = np.real(np.mean(desc_all))
    desc_std = np.real(np.std(desc_all))

    np.savez(
            outfile,
            desc_mean=desc_mean,
            desc_std=desc_std,
            )

def get_dir_path():
    """
    >>> get_dir_path()[-4:]
    'fep/'
    """
    return path.dirname(path.realpath(__file__)) + '/'

def get_samples_dir(dir_path, subdir):
    fpath = dir_path + subdir + '/'
    samples = list_of_directories(fpath)
    samples_dir = [fpath+sample for sample in samples]

    return samples_dir

def gen_dscribe_for(dir_path, phase):
    samples_dir = get_samples_dir(dir_path, phase)
    gen_dscribe(samples_dir, 6, 1)

def generate_training_set(dir_path):
    gen_dscribe_for(dir_path, 'training')
    samples_dir = get_samples_dir(dir_path, 'training')
    norm_descriptor(samples_dir, dir_path+'training/norm')

def generate_validation_set(dir_path):
    gen_dscribe_for(dir_path, 'prediction')

def main():
    dir_path = get_dir_path()

    generate_training_set(dir_path)
    generate_validation_set(dir_path)

if __name__ == "__main__":
    main()
