import subprocess

# Define parameters
plumed_file = 'knotted_8jrm.dat'
pdbfile = '/Users/sss/Documents/EnergyGap_project/positions_data/8jrm/8jrm_plumed.pdb'
dcdfile = '/Users/sss/Documents/EnergyGap_project/positions_data/8jrm/8jrm.dcd'

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
     