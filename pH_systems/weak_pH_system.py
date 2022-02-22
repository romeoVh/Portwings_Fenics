import numpy as np
from fenics import *

from numpy import linspace
from math import *
from mshr import *

class WeakPortHamiltonianSystem:
    "Class for a weak port Hamiltonian system with two states p,q."

    def __init__(self, V, problem, state_name):
        self.V = V
        #self.trialFn = TrialFunction(self.V)
        #self.testFn = TestFunction(self.V)
        # Define time variables
        self.t = Constant(0.0)  # Current time step
        self.t_1 = Constant(problem.dt)  # Next time step
        self.t_mid = Constant((problem.dt)/2.0)  # Mid time step

        # Define state variables at t and t+1
        self.state_t = Function(self.V, name=state_name)
        self.state_t_1 = Function(self.V, name=state_name + '+1')
        self.p_t = None
        self.q_t = None
        self.H_t = None

        # Exact State and Energy
        self.p_ex_t, self.q_ex_t = problem.get_exact_sol_at_t(self.t)
        self.H_ex_t = 0.5 * (inner(self.p_ex_t, self.p_ex_t) * dx(domain=problem.mesh) + inner(self.q_ex_t, self.q_ex_t) * dx(domain=problem.mesh))

        # Weak form LHS
        self.A = None

        # Boundary conditions
        self.bcArr = None


    def set_initial_condition(self,init_cond):
        self.state_t.assign(init_cond)
        self.p_t, self.q_t = self.state_t.split(deepcopy=True)
        print("Dimensions before and after: ", len(self.state_t.vector()), len(self.p_t.vector()), len(self.q_t.vector()))
        # Energy of System
        self.H_t = 0.5 * (inner(self.p_t, self.p_t) * dx + inner(self.q_t, self.q_t) * dx)

    def set_boundary_condition(self,problem,state_index):
        state_ex_t_1 = problem.get_exact_sol_at_t(self.t_1)
        self.bcArr = [DirichletBC(self.V.sub(state_index), state_ex_t_1[state_index], "on_boundary")]

    def outputs(self):
        # Calculates ||p_ex_t - p_t||,||q_ex_t - q_t||, H_t, H_ex_t
        err_p = errornorm(self.p_ex_t, self.p_t, norm_type="L2")
        err_q = errornorm(self.q_ex_t, self.q_t, norm_type="L2")
        H_ex_t = assemble(self.H_ex_t)
        H_t = assemble(self.H_t)
        return np.array([err_p,err_q,H_t,H_ex_t])


    def advance_time(self,dt):
        self.t.assign(float(self.t)+dt)
        self.t_1.assign(float(self.t_1) + dt)
        self.t_mid.assign(float(self.t_mid)+dt)

