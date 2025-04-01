import subprocess

# Define parameters
plumed_file = 'alignment_svd.dat'
pdbfile = '/Users/sss/Documents/EnergyGap_project/positions_data/1kdx_plumed.pdb'
dcdfile = '/Users/sss/Documents/EnergyGap_project/positions_data/1kdx.dcd'

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
     