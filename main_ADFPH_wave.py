from __future__ import print_function
from fenics import *
import numpy as np
from problems.rectangle_problem import *
from solvers.adaptive_dfpH_wave_solver import *

if __name__ == '__main__':
    options = None

    # Wave Problem 2D
    options = {"n_el": 1, "n_t": 5, "t_fin": 1.0}
    wave_rectangle = RectangleProblem(options)

    options = {"pol_deg":1,"bnd_cnd": "D","stagger_time":True,"couple_primal_dual":True}
    dfpHsolver = AdaptiveDualFieldPHWaveSolver(options)

    dfpHsolver.solve(wave_rectangle)






