import subprocess

# Define parameters
plumed_file = 'knotted_4dvd.dat'
pdbfile = '/Users/sss/Documents/EnergyGap_project/positions_data/4dvd/4dvd_plumed.pdb'
dcdfile = '/Users/sss/Documents/EnergyGap_project/positions_data/4dvd/4dvd.dcd'

# PLUMED command
command = [
    'plumed', 'driver',
    '--plumed', plumed_file,
    '--pdb', pdbfile,
    '--mf_dcd', dcdfile
]

# Executting
print(f"Running command: {' '.join(command)}")
subprocess.run(command)
     