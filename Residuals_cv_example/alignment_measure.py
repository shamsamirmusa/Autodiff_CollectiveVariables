import jax.numpy as jnp
import jax
from jax import grad, jit
import plumedCommunications as PLMD


plumedInit = {
    "COMPONENTS": {
        "Alighnemt": {"period": None, "derivative": True},}}

# @jit
def B_circumcenter(A, B, C):
   
    a =jnp.linalg.norm(B - C)
    b =jnp.linalg.norm( C - A)
    c =jnp.linalg.norm( A - B)

    alpha = (a**2)*(b**2 + c**2 - a**2)
    beta = (b**2)*(c**2 + a**2 - b**2)
    gamma = (c**2)*(a**2 + b**2 - c**2)
    
    denomenator = (alpha + beta + gamma)
    center_point = (alpha*A + beta*B + gamma*C )/ denomenator
    return center_point

# @jit
def circumcenter_matrix(positions):
   
    centers = jnp.zeros((len(positions)-2, 3))
    # distance_pace = jnp.zeros((len(positions)-2, 1))
    for i in range(len(positions) - 2):
        A, B, C = positions[i, :], positions[i+1, :], positions[i+2, :]
        cc = B_circumcenter(A, B, C)
        # jax.debug.print("{cc}", cc=cc)   
        centers = centers.at[i, :].set(cc)
        
    return  centers

def fit_line_to_circumcenters(allcenter):
    N = allcenter.shape[0]
    idxs = jnp.arange(N)

    def fitting_func(allcenter):
        A = jnp.vstack([idxs, jnp.ones(N)]).T
        m, c = jnp.linalg.lstsq(A,allcenter)[0]  #why -1 instead of none with jax
        return m, c
    
    m_x, c_x = fitting_func(allcenter[:,0]) 
    m_y, c_y = fitting_func(allcenter[:,1])
    m_z, c_z = fitting_func(allcenter[:,2])

    fitted_x = m_x*idxs + c_x
    fitted_y = m_y*idxs + c_y
    fitted_z = m_z*idxs + c_z

    residuals_x = allcenter[:,0] - fitted_x
    residuals_y = allcenter[:,1] - fitted_y
    residuals_z = allcenter[:,2] - fitted_z


    m = jnp.array([m_x, m_y, m_z])
    c = jnp.array([c_x, c_y,c_z])
    residuals = jnp.array([residuals_x, residuals_y, residuals_z])
    fitted_xyz = jnp.array([fitted_x, fitted_y, fitted_z])

    return m, c, residuals, fitted_xyz

def alignment(action: PLMD.PythonCVInterface):
    x= action.getPositions()
    def resid_fn(x):
        cm = circumcenter_matrix(x)
        _, _, residuals, _ = fit_line_to_circumcenters(cm)
        print(residuals)
        return residuals
    resid = resid_fn(x)
    # resid_grad  = grad(resid_fn)(x)
    # resid_grad_box = jnp.zeros((3,3))

    return resid# , resid_grad, resid_grad_box