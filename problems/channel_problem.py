
from problems.problem_base import *
import numpy as np

class ChannelProblem(ProblemBase):
    "2D channel test problem with known analytical solution?"

    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # Create mesh
        self.mesh = UnitSquareMesh(self.n_el, self.n_el)
        self.init_mesh(True)
        self.structured_time_grid()

        # Set viscosity (Re = 8)
        self.mu = 1.0 / 8.0
        # Set density
        self.rho = 1.0

    def initial_conditions(self, V_v, V_w, V_p):
        v_ex = Constant((0, 0))
        w_ex = Constant(0)
        p_ex = Expression("1 - x[0]",degree=3)

        v_init = interpolate(v_ex, V_v)
        if V_w is not None:
            w_init = interpolate(w_ex, V_w)
        else:
            w_init = None
        p_init = interpolate(p_ex, V_p)
        return [v_init, w_init, p_init]

    def boundary_conditions(self, V_v, V_w,V_p,t_c):
        bcw = []

        # Define boundaries
        inflow = 'near(x[0], 0)'
        outflow = 'near(x[0], 1)'
        walls = 'near(x[1], 0) || near(x[1], 1)'

        # Create no-slip boundary condition for velocity
        bcu_no_slip = DirichletBC(V_v, Constant((0.0, 0.0)), walls)

        # Create boundary conditions for pressure
        bcp_inflow = DirichletBC(V_p, Constant(1), inflow)
        bcp_outflow = DirichletBC(V_p, Constant(0), outflow)

        bcu = [bcu_no_slip]
        bcp = [bcp_inflow, bcp_outflow]

        return bcu,bcw, bcp

    def __str__(self):
        return "Channel"
