from __future__ import print_function
from problems.beltrami3d_problem import *
from solvers.dfpH_NS_solver import *
from solvers.ipcs_solver_new import *

if __name__ == '__main__':
    print("======== Besm Allah Al-Raman Al-Rahim ========")

    options = None

    # Beltrami 3D problem
    options = {"n_el":2,"n_t":100,"t_fin":1.0}
    beltrami = BeltramiProblem(options)

    options = {"pol_deg":2}
    ipcs = IPCS_Solver(options)
    ipcs.solve(beltrami)

    # options = {"pol_deg":1}
    # pH_NS = DualFieldPHNSSolver(options)
    # pH_NS.solve(beltrami)


    # Log book
    # divergence is zero only for pol_deg = 1