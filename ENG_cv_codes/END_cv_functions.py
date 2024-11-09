from collections import Counter

import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import matplotlib.pyplot as plt
import parmed as pmd
import re
import scipy.constants as sc
from statistics import covariance
import math
import matplotlib.cm as cm
import pandas as pd





# def compute_distance_matrix(u, residues):
#     """
#     Compute a matrix of distances between the centers of mass of the residues in the topology.

#     Parameters
#     ----------
#     u : MDAnalysis.Universe
#         The MDAnalysis universe containing the trajectory.
#     residues : list of MDAnalysis.core.groups.Residue
#         A list of residues in the universe.

#     Returns
#     -------
#     distance_matrix : np.ndarray
#         A symmetric matrix of distances between the centers of mass of the residues.
#     """
#     # Calculate the centers of mass of all residues at once
#     coms = np.array([residue.atoms.center_of_mass() for residue in residues])
#     # Compute the distance matrix using a vectorized calculation
#     distance_matrix = distances.distance_array(coms, coms, box=u.dimensions)
    
#     return distance_matrix
def compute_distance_matrix(u, residues):
    """
    Compute a matrix of distances between the centers of mass of the residues in the topology
    and log the residue identifiers.

    Parameters
    ----------
    u : MDAnalysis.Universe
        The MDAnalysis universe containing the trajectory.
    residues : list of MDAnalysis.core.groups.Residue
        A list of residues in the universe.

    Returns
    -------
    distance_matrix : np.ndarray
        A symmetric matrix of distances between the centers of mass of the residues.
    residue_identifiers : list
        List of identifiers corresponding to the residues in the distance matrix.
    """
    # Calculate the centers of mass of all residues at once
    coms = np.array([residue.atoms.center_of_mass() for residue in residues])
    # Log the residue identifiers
    residue_identifiers = [(residue.resid, residue.resname) for residue in residues]
    # Compute the distance matrix using a vectorized calculation
    distance_matrix = distances.distance_array(coms, coms, box=u.dimensions)
    
    return distance_matrix, residue_identifiers



def extract_lj_parameters(path):
    """
    #Extracts c6A, c12A, c6B, c12B from the topology file.
    Extract c6 and c12 from the topology file.

    Parameters:
    topology_file: str
        The path to the topology file in text format.

    Returns:
    lj_params: list of dicts
        A list of dictionaries with the extracted Lennard-Jones parameters for each functype.
    """
    with open(path, 'r') as file:
        data = file.read()

    # Regular expression patterns for LJ14 and LJ_SR types
    # lj14_pattern = r"c6A=\s*([0-9\.\-eE+]+),\s*c12A=\s*([0-9\.\-eE+]+),\s*c6B=\s*([0-9\.\-eE+]+),\s*c12B=\s*([0-9\.\-eE+]+)"
    lj_sr_pattern = r"c6=\s*([0-9\.\-eE+]+),\s*c12=\s*([0-9\.\-eE+]+)"

    # Find all LJ14 patterns
    #lj14_matches = re.findall(lj14_pattern, data)
    lj_sr_matches = re.findall(lj_sr_pattern, data)

    lj_params = []

    # Process LJ14 matches
    # for match in lj14_matches:
    #     c6A, c12A, c6B, c12B = match
    #     lj_params.append({
    #         "type": "LJ14",
    #         "c6A": float(c6A),
    #         "c12A": float(c12A),
    #         "c6B": float(c6B),
    #         "c12B": float(c12B)
    #     })

    # Process LJ_SR matches
    for match in lj_sr_matches:
        c6, c12 = match
        lj_params.append({
            "type": "LJ_SR",
            "c6": float(c6),
            "c12": float(c12)
        })

    return lj_params





def compute_interaction_energy_matrix(distance_matrix, residues, lj_params, dielectric=78.5):
    """
    Compute interaction energy matrix with Lennard-Jones and Coulombic interactions,
    adjusted for water as the dielectric medium and units in kJ/mol.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrix of distances between residues.
    residues : list
        List of residue objects.
    lj_params : dict
        Dictionary containing LJ parameters extracted from the topology file.
    dielectric : float
        Dielectric constant of the medium, default is 78.5 for water.

    Returns
    -------
    energy_matrix : np.ndarray
        Symmetric matrix of interaction energies.
    """
    n_residues = len(residues)
    energy_matrix = np.zeros((n_residues, n_residues))
    conversion_factor = 1 / (4 * np.pi * sc.epsilon_0 * dielectric) * sc.elementary_charge**2 / sc.Avogadro / 1000  # kJ/mol

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            E_vdw = 0.0
            E_coul = 0.0
            for atom_i in residues[i].atoms:
                for atom_j in residues[j].atoms:
                    r = np.linalg.norm(atom_i.position - atom_j.position)
                    if r > 0:
                        # Lennard-Jones parameters
                        type_i = atom_i.type
                        type_j = atom_j.type
                        if type_i in lj_params and type_j in lj_params:
                            c6_i = lj_params[type_i]["c6"]
                            c12_i = lj_params[type_i]["c12"]
                            c6_j = lj_params[type_j]["c6"]
                            c12_j = lj_params[type_j]["c12"]

                            # Compute sigma and epsilon
                            sigma_i = (c12_i / c6_i) ** (1 / 6)
                            epsilon_i = c6_i ** 2 / (4 * c12_i)
                            sigma_j = (c12_j / c6_j) ** (1 / 6)
                            epsilon_j = c6_j ** 2 / (4 * c12_j)

                            # Apply Lorentz-Berthelot mixing rules
                            sigma_ij = (sigma_i + sigma_j) / 2
                            epsilon_ij = np.sqrt(epsilon_i * epsilon_j)

                            # Lennard-Jones potential
                            E_vdw += 4 * epsilon_ij * ((sigma_ij / r)**12 - (sigma_ij / r)**6)

                        # Coulomb interaction
                        q_i = atom_i.charge
                        q_j = atom_j.charge
                        E_coul += conversion_factor * (q_i * q_j) / r

            # Total interaction energy
            energy_matrix[i, j] = E_vdw + E_coul
            energy_matrix[j, i] = energy_matrix[i, j]

    return energy_matrix



def calculate_eng_sdeng(eigenvalues):
    """
    Calculate ENG and SDENG from eigenvalues.
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    spectral_gap = sorted_eigenvalues[1] - sorted_eigenvalues[0]
    avg_separation = np.mean(np.diff(sorted_eigenvalues))
    eng = spectral_gap / avg_separation
    sdeng = np.std(sorted_eigenvalues)
    return eng, sdeng





def calculate_moving_average(data, window_size):
    """Compute the moving average using a simple sliding window approach."""
    if len(data) < window_size:
        return data  # Return the data itself if the window is too large
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'valid') 


def determine_optimal_window(data, min_window_size=1, max_percentage=11, increment_percentage=0.5):
    total_frames = len(data)
    max_window_size = min(int(max_percentage / 100 * total_frames), len(data) - 1)
    window_increment = max(int(increment_percentage / 100 * total_frames), 1)
    optimal_window_size = max(min_window_size, window_increment)  # Set a reasonable minimum window size

    previous_max = 0

    for window_size in range(optimal_window_size, max_window_size + 1, window_increment):
        smoothed_data = calculate_moving_average(data, window_size)
        current_max = np.max(smoothed_data)

        # Check if the current maximum ENG is less than or equal to the previous one to determine the optimal size
        if current_max <= previous_max:
            optimal_window_size = window_size - window_increment
            break

        previous_max = current_max

    return optimal_window_size



def determine_thresholds(eng_values, sdeng_values):
    eng_max = np.max(eng_values)
    sdeng_min = np.min(sdeng_values)
    msd = np.std(eng_values)  # Standard deviation of ENG values

    percentages = np.linspace(0, 100, 101)  # Generate percentages from 0 to 100
    eng_counts = []
    sdeng_counts = []

    for n in percentages:
        eng_threshold = eng_max - (n / 100) * msd
        sdeng_threshold = sdeng_min + (n / 100) * msd
        eng_count = np.sum(eng_values >= eng_threshold)
        sdeng_count = np.sum(sdeng_values <= sdeng_threshold)
        eng_counts.append(eng_count)
        sdeng_counts.append(sdeng_count)

   
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  

    # Plotting ENG counts on the first subplot
    ax[0].plot(percentages, eng_counts, label='ENG Counts', marker='o', color='b')
    ax[0].set_xlabel('Percentage of MSD from ENGmax')
    ax[0].set_ylabel('Number of Conformations')
    ax[0].set_title('ENG Threshold Determination')
    ax[0].legend()
    ax[0].grid(True)

    # Plotting SDENG counts on the second subplot
    ax[1].plot(percentages, sdeng_counts, label='SDENG Counts', marker='x', color='r')
    ax[1].set_xlabel('Percentage of MSD from ENGmax')
    ax[1].set_ylabel('Number of Conformations')
    ax[1].set_title('SDENG Threshold Determination')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()



    # Function to calculate the Pearson correlation coefficient
def calculate_pearson_correlation(eng, component):
    covariance = np.cov(eng, component)
    correlation_coefficient = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    return correlation_coefficient

# Function to calculate the correlation distance
def calculate_correlation_distance(rho):
    return np.sqrt(2 * (1 - rho))
