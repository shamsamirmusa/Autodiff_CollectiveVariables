import jax.numpy as jnp
import jax
from jax import jit
import plumedCommunications as PLMD

# plumedInit = {"Value": PLMD.defaults.COMPONENT} # ask toreturn a vector
plumedInit = {
    "COMPONENTS": {
        "length": {"period": None, "derivative": False},}}



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
    centers = jnp.zeros((len(positions)-2, 3))
    
    for i in range(len(positions) - 2):
        A, B, C = positions[i, :], positions[i+1, :], positions[i+2, :]
        cc = B_circumcenter(A, B, C)
        # jax.debug.print("{cc}", cc=cc)   
        centers = centers.at[i, :].set(cc)
        

    return  centers


def distance(action: PLMD.PythonCVInterface):
    x= action.getPositions()
    cm = circumcenter_matrix(x)
    dist = jnp.linalg.norm(cm[0, :] - cm[-1, :])
    print(dist)
    
    return dist


    
    