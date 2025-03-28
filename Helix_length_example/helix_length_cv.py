import jax.numpy as jnp
import jax
from jax import jit
jax.config.update("jax_enable_x64", True) 
import plumedCommunications as PLMD


# plumedInit = {"Value": PLMD.defaults.COMPONENT} # ask toreturn a vector
plumedInit = {
    "COMPONENTS": {
        "length": {"period": None, "derivative": True},}}



@jit
def B_circumcenter(A, B, C):

    """
    The function returns the circumcenter of a triangle ABC based on the barycentric coordinates explained in the following link
    https://math.stackexchange.com/questions/1304194/center-of-circle-when-three-points-in-3-space-are-given/1304202#1304202

    Parameters
    ----------
    A : numpy array
        coordinates of the first point in the triangle.
    B : numpy array
        coordinates of the second point in the triangle
    C : numpy array
        coordinates of the third point in the triangle

    Returns
    -------
    center_point : numpy array
        coordinates of the circumcenter
    
    """
   
    a =jnp.linalg.norm(B - C)
    b =jnp.linalg.norm( C - A)
    c =jnp.linalg.norm( A - B)

    alpha = (a**2)*(b**2 + c**2 - a**2)
    beta = (b**2)*(c**2 + a**2 - b**2)
    gamma = (c**2)*(a**2 + b**2 - c**2)
    
    denomenator = (alpha + beta + gamma)
    center_point = (alpha*A + beta*B + gamma*C )/ denomenator
    return center_point


@jit
def circumcenter_matrix(positions):
    """
    This function takes a (N, 3) array of positions of N atoms in 3D space and returns a (N-2, 3) array of the circumcenters of all consecutive triplets of atoms
    
    Parameters
    ----------
    positions : numpy array
        (N, 3) array of positions of N atoms in 3D space
    
    Returns
    -------
    centers : numpy array
        (N-2, 3) array of the circumcenters of all consecutive triplets of atoms
    """
@jit
def circumcenter_matrix(positions):
    N = positions.shape[0]
    indices = jnp.arange(N - 2)
    
    def compute_cc(i):
        A = positions[i]
        B = positions[i + 1]
        C = positions[i + 2]
        return B_circumcenter(A, B, C)
    
    # Apply the compute_cc function to every index using vmap.
    center = jax.vmap(compute_cc)(indices)
    return center

@jit
def helper_distance(allcenters):
    dist = jnp.linalg.norm(allcenters[0, :] - allcenters[-1, :])
    return dist

def distance_from_positions(positions):
    cm = circumcenter_matrix(positions)
    return helper_distance(cm)

gradient = jit(jax.grad(distance_from_positions))


def distance(action: PLMD.PythonCVInterface):
    
    x= action.getPositions()
    cm = circumcenter_matrix(x)
    dist = helper_distance(cm)
    grad_d = gradient(x)
    dummy_box = jnp.zeros((3,3))
    return dist, grad_d,dummy_box


    
    