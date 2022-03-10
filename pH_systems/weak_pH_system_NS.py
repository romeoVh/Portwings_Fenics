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
        self.E_t = None
        self.Ch_t = None
        self.div_v_t = None

        # Exact State and Energy (Moved to problem class - e.g. Beltrami)
        #self.v_ex_t, self.w_ex_t , self.p_ex_t = problem.get_exact_sol_at_t(self.t)
        #self.H_ex_t = 0.5 * (inner(self.v_ex_t, self.v_ex_t) * dx(domain=problem.mesh))
        self.prob_output_arr = None

        # Weak form LHS
        self.A = None

        # Boundary conditions
        self.bcArr = None


    def set_initial_condition(self,init_cond):
        self.state_t.assign(init_cond)
        self.v_t, self.w_t, self.p_t = self.state_t.split(deepcopy=True)
        # print("Dimensions before and after: ", len(self.state_t.vector()), len(self.v_t.vector()), len(self.w_t.vector()), len(self.p_t.vector()))

    def init_outputs(self):
        # 4 outputs:
        # Energy, Enstrophy and Helicity of System and divergence of velocity vector field
        self.H_t = 0.5 * inner(self.v_t, self.v_t) * dx
        self.E_t = 0.5 * inner(self.w_t, self.w_t) * dx
        self.Ch_t = dot(self.v_t, self.w_t)  * dx
        self.div_v_t = (div(self.v_t)) ** 2 * dx
        return 4

    def set_boundary_condition_old(self,problem,state_index,sub_domain):
        state_ex_t_1 = problem.get_exact_sol_at_t(self.t_1)
        print(state_ex_t_1[0])
        x = DirichletBC(self.V.sub(state_index), state_ex_t_1[state_index], sub_domain)
        if(self.bcArr is None):
            self.bcArr = [x]
        else:
            self.bcArr.append(x)
        pass

    def set_boundary_condition(self,bc):
        if (self.bcArr is None):
            self.bcArr = [bc]
        else:
            self.bcArr.append(bc)
        pass

    def outputs(self,problem):
        # Calculates problem specific outputs in addition to H_t & ||div(u_t)||
        prob_out = problem.calculate_outputs(self.prob_output_arr,self.v_t,self.w_t,self.p_t)
        div_v = assemble(self.div_v_t)
        H_t = assemble(self.H_t)
        E_t = assemble(self.E_t)
        Ch_t = assemble(self.Ch_t)

        return np.append(prob_out,np.array([H_t,E_t,Ch_t,div_v]))


    def advance_time(self,dt):
        self.t.assign(float(self.t)+dt)
        self.t_1.assign(float(self.t_1) + dt)
        self.t_mid.assign(float(self.t_mid)+dt)

    def time_march(self,b_form,dt,solver,precond):
        b_mat = assemble(b_form)

        if not (self.bcArr is None): [bc.apply(self.A, b_mat) for bc in self.bcArr]
        solve(self.A, self.state_t_1.vector(), b_mat, solver, precond)

        v_k_1, w_k_1, p_k_1 = self.state_t_1.split(deepcopy=True)

        self.v_t.assign(v_k_1)
        self.w_t.assign(w_k_1)
        self.p_t.assign(p_k_1)

        # Advance time step
        self.advance_time(dt)

