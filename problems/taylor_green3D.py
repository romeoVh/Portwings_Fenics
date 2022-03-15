
from problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym

class TaylorGreen3D(ProblemBase):
    "3D Taylor Green problem."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        self.mesh = BoxMesh(Point(-pi,-pi, -pi), Point(pi, pi, pi), self.n_el, self.n_el, self.n_el)
        self.init_mesh()
        self.structured_time_grid()

        # Set viscosity
        self.mu = 1.0 / 500
        # Set density
        self.rho = 1
        # Reynolds number
        self.Re = self.rho / self.mu
        # Periodic problem
        self.periodic = True
        # Solution is not exact
        self.exact = False

    def initial_conditions(self, V_v, V_w, V_p):

        v_0 = ("sin(x[0])*cos(x[1])*cos(x[2])", "-cos(x[0])*sin(x[1])*cos(x[2])", "0")
        v_ex_0 = Expression(v_0, degree=2)
        v_init = interpolate(v_ex_0, V_v)

        w_0 = ("-cos(x[0])*sin(x[1])*sin(x[2])", "-sin(x[0])*cos(x[1])*sin(x[2])", "2*sin(x[0])*sin(x[1])*cos(x[2])")
        w_ex_0 = Expression(w_0, degree=2)
        w_init = interpolate(w_ex_0, V_w)

        p_0 = "1/16*(cos(2*x[0]) + cos(2*x[1]))*(cos(2*x[2]) + 2)"
        p_ex_0 = Expression(p_0, degree=2)
        p_init = interpolate(p_ex_0, V_p)
        return [v_init, w_init, p_init]

    def boundary_conditions(self, V_v, V_w,V_p,t_c):
        # Periodic boundary conditions are simply empty
        bcu = []
        bcw = []
        bcp = []

        return bcu,bcw, bcp


    def __str__(self):
        return "TaylorGreen3D"