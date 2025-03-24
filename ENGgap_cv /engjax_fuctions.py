import jax.numpy as jnp
from jax.scipy.linalg import eigh
from jax import grad
import scipy.constants as sc
import numpy as np
import MDAnalysis as mda
import re



# Compute Interaction Energy Matrix
def compute_interaction_energy_matrix_flat(coords, residues, lj_params, dielectric=78.5):
    """
    Computes the interaction energy matrix using JAX and vectorized operations.
    """
    n_residues = len(residues)
    energy_matrix = jnp.zeros((n_residues, n_residues))

    conversion_factor = (
        1 / (4 * jnp.pi * sc.epsilon_0 * dielectric)
        * sc.elementary_charge**2
        / sc.Avogadro
        / 1000
    )  # kJ/mol

    # Residue-wise energy matrix calculation
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            e_vdw_total, e_coul_total = 0.0, 0.0
            for atom_i in residues[i].atoms:
                for atom_j in residues[j].atoms:
                    pos_i = atom_i.position
                    pos_j = atom_j.position
                    r = jnp.linalg.norm(pos_i - pos_j)

                    if r > 0:
                        # Lennard-Jones parameters
                        type_i, type_j = atom_i.type, atom_j.type
                        if type_i in lj_params and type_j in lj_params:
                            c6_i, c12_i = lj_params[type_i]["c6"], lj_params[type_i]["c12"]
                            c6_j, c12_j = lj_params[type_j]["c6"], lj_params[type_j]["c12"]

                            sigma_i = (c12_i / c6_i) ** (1 / 6)
                            epsilon_i = c6_i**2 / (4 * c12_i)
                            sigma_j = (c12_j / c6_j) ** (1 / 6)
                            epsilon_j = c6_j**2 / (4 * c12_j)

                            sigma_ij = (sigma_i + sigma_j) / 2
                            epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)

                            e_vdw = 4 * epsilon_ij * ((sigma_ij / r)**12 - (sigma_ij / r)**6)
                        else:
                            e_vdw = 0.0

                        q_i, q_j = atom_i.charge, atom_j.charge
                        e_coul = conversion_factor * (q_i * q_j) / r

                        e_vdw_total += e_vdw
                        e_coul_total += e_coul

            energy_matrix = energy_matrix.at[i, j].set(e_vdw_total + e_coul_total)
            energy_matrix = energy_matrix.at[j, i].set(e_vdw_total + e_coul_total)

    return energy_matrix




def extract_lj_parameters(lj_parameters_file ):
    """
    Extracts c6 and c12 from the topology file using regex.
    """
    with open(lj_parameters_file , 'r') as file:
        data = file.read()

    # Regular expression patterns for LJ_SR types
    lj_sr_pattern = r"c6=\s*([0-9\.\-eE+]+),\s*c12=\s*([0-9\.\-eE+]+)"
    lj_sr_matches = re.findall(lj_sr_pattern, data)

    lj_params = []
    for match in lj_sr_matches:
        c6, c12 = match
        lj_params.append({
            "type": "LJ_SR",
            "c6": float(c6),
            "c12": float(c12)
        })

    return lj_params




# Calculate ENG(t), Eigenvalues, and Eigenvectors
def calculate_eng_t_flat(coords, residues, lj_params):
    interaction_matrix = compute_interaction_energy_matrix_flat(coords, residues, lj_params)
    eigenvalues, eigenvectors = eigh(interaction_matrix)
    eigenvalues = jnp.sort(eigenvalues)
    spectral_gap = eigenvalues[1] - eigenvalues[0]
    avg_separation = jnp.mean(jnp.diff(eigenvalues))
    eng_t = spectral_gap / avg_separation
    sdeng = jnp.std(eigenvalues)
    return eng_t, eigenvalues, eigenvectors, sdeng




# Compute Residue Time Series
def compute_residue_time_series(u, protein_residues, lj_params):
    residue_time_series = {residue.resname + "_" + str(residue.resid): [] for residue in protein_residues}
    for ts in u.trajectory:
        coords = np.array([atom.position for atom in protein_residues.atoms])
        coords_flat = jnp.array(coords.reshape(-1, 3))
        _, _, eigenvectors = calculate_eng_t_flat(coords_flat, protein_residues, lj_params)
        first_eigenvector = eigenvectors[:, 0]
        start_idx = 0
        for residue in protein_residues:
            num_atoms = len(residue.atoms)
            residue_contribution = jnp.sum(first_eigenvector[start_idx:start_idx + num_atoms])
            residue_time_series[residue.resname + "_" + str(residue.resid)].append(residue_contribution)
            start_idx += num_atoms
    return residue_time_series

# Compute Correlation Distance (CD)
def compute_cd(eng_time_series, residue_time_series):
    eng_time_series = jnp.array(eng_time_series)
    residue_time_series = jnp.array(residue_time_series)
    covariance = jnp.cov(eng_time_series, residue_time_series)[0, 1]
    std_x = jnp.std(eng_time_series)
    std_y = jnp.std(residue_time_series)
    rho = covariance / (std_x * std_y)
    cd = jnp.sqrt(2 * (1 - rho))
    return cd

# Fine-Tuning Loop 2
def fine_tuning_loop_2(eng_time_series, residue_time_series, alpha, beta):
    cd = compute_cd(eng_time_series, residue_time_series)
    SQRT_2 = jnp.sqrt(2)
    if cd > SQRT_2:
        alpha += 0.1  # Increase alpha for better exploration
    else:
        beta += 0.1  # Increase beta to reduce fluctuations
    return alpha, beta

# Coarse-Tuning Loop
def coarse_tuning_loop(eng_time_series, sdeng_time_series, percentages):
    eng_max = max(eng_time_series)
    sdeng_min = min(sdeng_time_series)
    msd = jnp.std(eng_time_series)
    eng_counts, sdeng_counts, folded_probabilities = [], [], []
    for n in percentages:
        eng_threshold = eng_max - (n / 100) * msd
        sdeng_threshold = sdeng_min + (n / 100) * msd
        eng_count = sum(1 for eng in eng_time_series if eng > eng_threshold)
        sdeng_count = sum(1 for sdeng in sdeng_time_series if sdeng < sdeng_threshold)
        folded_probability = (eng_count + sdeng_count) / (2 * len(eng_time_series))
        eng_counts.append(eng_count)
        sdeng_counts.append(sdeng_count)
        folded_probabilities.append(folded_probability)
    return folded_probabilities, eng_counts, sdeng_counts

# Process Trajectory with Coarse Tuning
def process_trajectory_with_coarse_tuning(u, protein_residues, lj_params, percentages):
    eng_time_series, sdeng_time_series = [], []
    alpha, beta = 1.0, 1.0
    for ts in u.trajectory:
        coords = np.array([atom.position for atom in protein_residues.atoms])
        coords_flat = jnp.array(coords.reshape(-1, 3))
        eng_t, eigenvalues, _ = calculate_eng_t_flat(coords_flat, protein_residues, lj_params)
        eng_time_series.append(eng_t)
        sdeng_time_series.append(jnp.std(eigenvalues))
        if len(eng_time_series) > 1:
            alpha, beta = fine_tuning_loop_2(
                eng_time_series[-2:], sdeng_time_series[-2:], alpha, beta
            )
    folded_probabilities, eng_counts, sdeng_counts = coarse_tuning_loop(
        eng_time_series, sdeng_time_series, percentages
    )
    return eng_time_series, sdeng_time_series, folded_probabilities, eng_counts, sdeng_counts, alpha, beta



import pandas as pd

# Function to calculate Pearson correlation coefficient
def calculate_pearson_correlation(eng_list, residue_values):
    """
    Calculate Pearson correlation coefficient between ENG(t) and residue contributions.
    """
    eng_array = jnp.array(eng_list)
    residue_array = jnp.array(residue_values)
    covariance = jnp.cov(eng_array, residue_array)[0, 1]
    std_eng = jnp.std(eng_array)
    std_residue = jnp.std(residue_array)
    rho = covariance / (std_eng * std_residue)
    return rho

# Function to calculate Correlation Distance (CD)
def calculate_correlation_distance(rho):
    """
    Calculate the Correlation Distance (CD) from the Pearson correlation coefficient.
    """
    return jnp.sqrt(2 * (1 - rho))

# Process residue correlations
def process_residue_correlations(residue_time_series, eng_time_series):
    """
    Calculate the correlation coefficient and correlation distance for each residue.
    """
    residue_correlations = {}

    for residue, values in residue_time_series.items():
        rho = calculate_pearson_correlation(eng_time_series, values)
        cd = calculate_correlation_distance(rho)
        residue_correlations[residue] = {
            'Correlation Coefficient': rho.item(),  # Convert JAX array to Python scalar
            'Correlation Distance': cd.item()
        }

    # Sort residues by Correlation Distance
    sorted_residues = sorted(
        residue_correlations.items(),
        key=lambda item: item[1]['Correlation Distance']
    )

    # Convert to a DataFrame for analysis
    correlation_df = pd.DataFrame({
        'Residue': [res[0] for res in sorted_residues],
        'Correlation Coefficient': [res[1]['Correlation Coefficient'] for res in sorted_residues],
        'Correlation Distance': [res[1]['Correlation Distance'] for res in sorted_residues]
    })

    return correlation_df
