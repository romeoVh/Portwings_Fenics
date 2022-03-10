
from problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym

class DrivenCavityProblem(ProblemBase):
    "2D lid-driven cavity test problem with known reference value."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # Create space-time mesh
        # We start with a UnitCube and modify it to get the mesh we want: (-1, 1) x (-1, 1) x (-1, 1)
        self.mesh = UnitSquareMesh(self.n_el, self.n_el)
        self.init_mesh()
        self.structured_time_grid()

        # Set viscosity
        self.mu = 1.0 / 1000.0
        # Set density
        self.rho = 1.0

    def initial_conditions(self, V_v, V_w, V_p):
        v_init = Constant((0.0, 0.0))
        w_init = Constant(0.0)
        p_init = Constant(0.0)
        v_init = interpolate(v_init, V_v)
        if V_w is not None: w_init = interpolate(w_init, V_w)
        p_init = interpolate(p_init, V_p)
        return [v_init,w_init,p_init]

    def boundary_conditions(self, V_v, V_w,V_p,t_c):
        bcu = []
        bcw = []
        bcp = []

        # Define boundaries
        parallel_flow = 'near(x[1], 1)'
        walls = 'near(x[0], 0) || near(x[0], 1) || near(x[1], 0)'

        # Create no-slip boundary condition for velocity
        bcu_no_slip = DirichletBC(V_v, Constant((0.0, 0.0)), walls)
        bcu_flow = DirichletBC(V_v, Constant((1.0, 0.0)), parallel_flow)

        bcu = [bcu_no_slip,bcu_flow]

        return bcu,bcw, bcp

    def __str__(self):
        return "DrivenCavity"