import numpy as np
from causal_kv.ode_solvers import local_solution,coupled_solution,accuracy_audit


def test_analytic_cases():
    t=np.linspace(0,1,64)
    a,k,z,u=.7,.2,.4,.6
    reference=z*np.exp(-a*t)+k*u*(-np.expm1(-a*t))/a
    np.testing.assert_allclose(local_solution(a,k,z,np.full(64,u),t),reference,atol=1e-13)
    full=coupled_solution(np.array([[a,1.1,0,0]]),np.array([[z,-.3]]),t)
    np.testing.assert_allclose(full[0,:,0],z*np.exp(-a*t),atol=1e-13)
    np.testing.assert_allclose(full[0,:,1],-.3*np.exp(-1.1*t),atol=1e-13)


def test_numerical_audit():
    audit=accuracy_audit()
    assert audit['linear_forcing_solver_max_error']<1e-9
    assert audit['matrix_exponential_max_error']<1e-9
    errs=[r['smooth_forcing_max_error'] for r in audit['grid_convergence']]
    assert all(a/b>3.5 for a,b in zip(errs,errs[1:]))
