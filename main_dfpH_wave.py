from __future__ import print_function
from fenics import *
import numpy as np
from box_problem import *
from dfpH_wave_solver import *

if __name__ == '__main__':
    print("======== Besm Allah Al-Raman Al-Rahim ========")

    options = None

    # Wave 3D problem
    options = {"n_el":2,"n_t":50,"t_fin":5}
    wave_box = BoxProblem(options)

    options = {"pol_deg":2,"bnd_cnd": "D","stagger_time":True,"couple_primal_dual":False}
    dfpHsolver = DualFieldPHWaveSolver(options)

    dfpHsolver.solve(wave_box)







