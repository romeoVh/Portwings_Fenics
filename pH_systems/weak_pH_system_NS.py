import numpy as np
from fenics import *

from numpy import linspace
from math import *
from mshr import *

class WeakPortHamiltonianSystemNS:
    "Class for a weak port Hamiltonian system of the incompresible Navier Stokes equation."

    def __init__(self, V, problem, state_name):
        self.V = V
        # Define time variables
        self.t = Constant(0.0)  # Current time step
        self.t_1 = Constant(problem.dt)  # Next time step
        self.t_mid = Constant((problem.dt)/2.0)  # Mid time step

        # Define state variables at t and t+1
        self.state_t = Function(self.V, name=state_name)
        self.state_t_1 = Function(self.V, name=state_name + '+1')
        self.v_t = None
        self.w_t = None
        self.p_t = None
        self.H_t = None
        self.div_v_t = None

        # Exact State and Energy
        self.v_ex_t, self.w_ex_t , self.p_ex_t = problem.get_exact_sol_at_t(self.t)
        self.H_ex_t = 0.5 * (inner(self.v_ex_t, self.v_ex_t) * dx(domain=problem.mesh))

        # Weak form LHS
        self.A = None

        # Boundary conditions
        self.bcArr = None


    def set_initial_condition(self,init_cond):
        self.state_t.assign(init_cond)
        self.v_t, self.w_t, self.p_t = self.state_t.split(deepcopy=True)
        # print("Dimensions before and after: ", len(self.state_t.vector()), len(self.v_t.vector()), len(self.w_t.vector()), len(self.p_t.vector()))
        # Energy of System and divergence of velocity vector field
        self.H_t = 0.5 * inner(self.v_t, self.v_t) * dx
        self.div_v_t = (div(self.v_t))**2* dx



    def set_boundary_condition(self,problem,state_index,sub_domain):
        state_ex_t_1 = problem.get_exact_sol_at_t(self.t_1)
        x = DirichletBC(self.V.sub(state_index), state_ex_t_1[state_index], sub_domain)
        if(self.bcArr is None):
            self.bcArr = [x]
        else:
            self.bcArr.append(x)

    def outputs(self):
        # Calculates ||v_ex_t - v_t||,||w_ex_t - w_t||,||p_ex_t - p_t||,||div(u_t)||, H_t, H_ex_t
        err_v = errornorm(self.v_ex_t, self.v_t, norm_type="L2")
        err_w = errornorm(self.w_ex_t, self.w_t, norm_type="L2")
        err_p = errornorm(self.p_ex_t, self.p_t, norm_type="L2")
        div_v = assemble(self.div_v_t)
        H_t = assemble(self.H_t)
        H_ex_t = assemble(self.H_ex_t)
        return np.array([err_v,err_w,err_p,div_v,H_t,H_ex_t])


    def advance_time(self,dt):
        self.t.assign(float(self.t)+dt)
        self.t_1.assign(float(self.t_1) + dt)
        self.t_mid.assign(float(self.t_mid)+dt)

