import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD

plumedInit = {
    "COMPONENTS": {
        "straight": {"period": None, "derivative": True},
        "dist": {"period": None, "derivative": True}
    }
}

@jit
def B_circumcenter(A, B, C):
    a = jnp.linalg.norm(B - C)
    b = jnp.linalg.norm(C - A)
    c = jnp.linalg.norm(A - B)
    alpha = (a**2)*(b**2 + c**2 - a**2)
    beta  = (b**2)*(c**2 + a**2 - b**2)
    gamma = (c**2)*(a**2 + b**2 - c**2)
    denominator = alpha + beta + gamma
    return (alpha*A + beta*B + gamma*C) / denominator

@jit
def circumcenter_matrix(positions):
    # positions shape: (natoms, 3)
    n_triplets = len(positions) - 2
    centers = jnp.zeros((n_triplets, 3))
    for i in range(n_triplets):
        A = positions[i]
        B = positions[i+1]
        C = positions[i+2]
        cc = B_circumcenter(A, B, C)
        centers = centers.at[i].set(cc)
    return centers

def svd_straightness(points):
    """Compute ratio of largest singular value to sum of singular values."""
    avg = jnp.mean(points, axis=0)
    std = jnp.std(points, axis=0) + 1e-9  # avoid /0
    centered = (points - avg) / std
    _, s, _ = jnp.linalg.svd(centered, full_matrices=False)
    return s[0] / (jnp.sum(s) + 1e-9)

def measure_straightness(x):
    c_points = circumcenter_matrix(x)
    return svd_straightness(c_points)

def measure_dist(x):
    """Distance between first and last circumcenters."""
    c_points = circumcenter_matrix(x)
    return jnp.linalg.norm(c_points[0] - c_points[-1])

# Gradients of each measure w.r.t. original atomic positions
grad_straightness = grad(measure_straightness)
grad_dist = grad(measure_dist)

def distance(action: PLMD.PythonCVInterface):
    
    x = action.getPositions()  # shape: (natoms, 3)

    # Compute values
    val_dist = measure_dist(x)

    # Compute gradients w.r.t. x # shape (natoms,3)
    grad_dist_val = grad_dist(x)              # shape (natoms,3)

    # Dummy 3x3 boxes to match PythonCVInterface signature
    
    dummy_box2 = jnp.zeros((3,3))

    return val_dist, grad_dist_val, dummy_box2
   

    # Return structure for 2-component CV:
    # [value1, value2], [grad1, grad2], [box1, box2]
    #return val_straight, val_dist, grad_straight_val, grad_dist_val, dummy_box1, dummy_box2
    

def alignment(action: PLMD.PythonCVInterface):
    
    x = action.getPositions()  # shape: (natoms, 3)

    # Compute values
    val_straight = measure_straightness(x)

    # Compute gradients w.r.t. x
    grad_straight_val = grad_straightness(x)  # shape (natoms,3)
             # shape (natoms,3)
    # Dummy 3x3 boxes to match PythonCVInterface signature
    dummy_box1 = jnp.zeros((3,3))


    return val_straight, grad_straight_val, dummy_box1