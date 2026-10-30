"""Float64 references. Local forcing is defined by linear interpolation of samples."""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def local_solution(decay, coupling, initial, forcing, time):
    """Exact integrating-factor step for piecewise-linear forcing; arbitrary batch axes."""
    u = np.asarray(forcing, dtype=np.float64)
    a, k, z = np.broadcast_arrays(decay, coupling, initial)
    out = np.empty_like(u)
    out[..., 0] = z
    for n, h in enumerate(np.diff(time)):
        e = np.exp(-a*h)
        c0 = -np.expm1(-a*h)/a
        c1 = (h-c0)/a
        z = e*z + k*(c0*u[..., n]+c1*(u[..., n+1]-u[..., n])/h)
        out[..., n+1] = z
    return out


def coupled_solution(parameters, initial, time):
    """Batched matrix exponential reference for [a,b,kappa,lambda]."""
    p = np.asarray(parameters)
    matrices = np.zeros((len(p), 2, 2), dtype=np.float64)
    matrices[:, 0, 0], matrices[:, 1, 1] = -p[:, 0], -p[:, 1]
    matrices[:, 0, 1], matrices[:, 1, 0] = p[:, 2], p[:, 3]
    # scipy's batched expm; evaluates the original coupled ODE, not relaxation.
    e = expm(np.asarray(time)[None, :, None, None]*matrices[:, None])
    return np.einsum('btij,bj->bti', e, initial)


def accuracy_audit(seed=913):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 64)
    u = rng.normal(size=64)
    a, k, z = .83, -.21, .45
    ref = local_solution(a, k, z, u, t)
    # Integrate each linear segment so adaptive integration cannot skip knots.
    numeric = [z]
    for i in range(63):
        h = t[i+1]-t[i]
        sol = solve_ivp(lambda s, x: -a*x+k*(u[i]+(u[i+1]-u[i])*(s-t[i])/h),
                        (t[i], t[i+1]), [numeric[-1]], rtol=1e-11, atol=1e-13)
        numeric.append(sol.y[0, -1])
    errors = []
    for n in (32, 64, 128, 256):
        tn = np.linspace(0, 1, n)
        un = np.sin(2*np.pi*tn)
        xn = local_solution(a, k, z, un, tn)
        exact = solve_ivp(lambda s, x: -a*x+k*np.sin(2*np.pi*s), (0, 1), [z],
                          t_eval=tn, rtol=1e-12, atol=1e-14).y[0]
        errors.append({'points': n, 'smooth_forcing_max_error': float(np.max(abs(xn-exact)))})
    p = np.array([[.6, 1.4, -.25, .25], [1.2, .7, 1.5, 1.5]])
    initial = np.array([[.4, -.7], [.2, .8]])
    full = coupled_solution(p, initial, t)
    maxerr = 0.
    for i, (aa, bb, kk, ll) in enumerate(p):
        matrix = np.array([[-aa, kk], [ll, -bb]])
        numerical = solve_ivp(lambda s, q: matrix@q, (0, 1), initial[i], t_eval=t,
                              rtol=1e-12, atol=1e-14).y.T
        maxerr = max(maxerr, float(np.max(abs(full[i]-numerical))))
    return {'linear_forcing_solver_max_error': float(np.max(abs(ref-numeric))),
            'matrix_exponential_max_error': maxerr, 'grid_convergence': errors}
