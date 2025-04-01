import subprocess

# Define parameters
plumed_file = 'knotted_7sq3.dat'
pdbfile = '/Users/sss/Documents/EnergyGap_project/positions_data/7sq3/7sq3_plumed.pdb'
dcdfile = '/Users/sss/Documents/EnergyGap_project/positions_data/7sq3/7sq3.dcd'

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
     