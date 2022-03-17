
from problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym

class ConservationProperties3D(ProblemBase):
    "3D Taylor Green problem."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        self.mesh = BoxMesh(Point(0,0,0), Point(1, 1, 1), self.n_el, self.n_el, self.n_el)
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

        v_0 = ("cos(2*pi*x[2])", "sin(2*pi*x[2])", "sin(2*pi*x[0])")
        v_ex_0 = Expression(v_0, degree=2)
        v_init = interpolate(v_ex_0, V_v)

        w_0 = ("-2*pi*cos(2*pi*x[2])", "-2*pi*(cos(2*pi*x[0]) + sin(2*pi*x[2]))", "0")
        w_ex_0 = Expression(w_0, degree=2)
        w_init = interpolate(w_ex_0, V_w)

        p_0 = "0"
        p_ex_0 = Expression(p_0, degree=2)
        p_init = interpolate(p_ex_0, V_p)
        return [v_init, w_init, p_init]

    def boundary_conditions(self):
        pbc = PeriodicBoundary()

        return pbc

    def __str__(self):
        return "ConservationProperties3D"

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool ((near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and
            (not ((near(x[0], 1) and near(x[2], 1)) or
                  (near(x[0], 1) and near(x[1], 1)) or
                  (near(x[1], 1) and near(x[2], 1)))) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
    	#### define mapping for a single point in the box, such that 3 mappings are required
        if near(x[0], 1) and near(x[1], 1) and near(x[2], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
            y[2] = x[2] - 1
        ##### define mapping for edges in the box, such that mapping in 2 Cartesian coordinates are required
        if near(x[0], 1) and near(x[2], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            y[2] = x[2] - 1
        elif near(x[1], 1) and near(x[2], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
            y[2] = x[2] - 1
        elif near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
            y[2] = x[2]
        #### right maps to left: left/right is defined as the x-direction
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            y[2] = x[2]
        ### back maps to front: front/back is defined as the y-direction
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
            y[2] = x[2]
        #### top maps to bottom: top/bottom is defined as the z-direction
        elif near(x[2], 1):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 1
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000