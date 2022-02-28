from fenics import *
from time import time
from tqdm import tqdm
from solvers.solver_base import *
from pH_systems.weak_pH_system_NS import *

class DualFieldPHNSSolver(SolverBase):
    # "Dual Field port-Hamiltonian scheme for Navier-Stokes equation."
    def __init__(self, options):
        SolverBase.__init__(self, options)
        # Test functions primal and dual
        self.chi_1 = self.chi_2 = self.chi_0 = None
        self.chiT_n1 = self.chiT_n2 = self.chiT_n = None
        # Trial functions primal and dual
        self.v = self.w = self.p = None
        self.vT = self.wT = self.pT = None

        self.stagger_time = options["stagger_time"]
        self.couple_primal_dual = options["couple_primal_dual"]


    def __str__(self):
        return "DFPH_NS_3D"

    def init_test_trial_functions(self,V_primal,V_dual):
        chi_primal = TestFunction(V_primal)
        chi_dual = TestFunction(V_dual)
        self.chi_1 , self.chi_2 , self.chi_0  = split(chi_primal)
        self.chiT_n1 , self.chiT_n2 , self.chiT_n  = split(chi_dual)

        # Define Unknown Trial functions
        x_primal = TrialFunction(V_primal)
        x_dual = TrialFunction(V_dual)
        self.v , self.w , self.p  = split(x_primal)
        self.vT , self.wT , self.pT  = split(x_dual)

    def assemble_lhs_primal(self,dt,pH_P,input_n2):
        a_form_eq1 = (1/dt) * m_i(self.chi_1, self.v) - 0.5 * eta_s(self.chi_1, self.v, input_n2) \
                     - 0.5 * eta_p(self.chi_1, self.p) - 0.5 * eta_k(self.chi_1, self.w, self.kappa)
        a_form_eq2 = -0.5 * eta_p_Tr(self.chi_0, self.v)
        a_form_eq3 = m_i(self.chi_2, self.w) - eta_k_Tr(self.chi_2, self.v)
        pH_P.A = assemble(a_form_eq1+a_form_eq2+a_form_eq3)

    def assemble_lhs_dual(self,dt,pH_D,input_2):
        a_form_eq1 = (1 / dt) * m_i(self.chiT_n1, self.vT) - 0.5 * etaT_s(self.chiT_n1, self.vT, input_2) \
                     - 0.5 * etaT_p(self.chiT_n1, self.pT) - 0.5 * etaT_k(self.chiT_n1, self.wT, self.kappa)
        a_form_eq2 = etaT_p_Tr(self.chiT_n, self.vT)
        a_form_eq3 = 0.5 * m_i(self.chiT_n2, self.wT) - 0.5 * etaT_k_Tr(self.chiT_n2, self.vT)
        pH_D.A = assemble(a_form_eq1 + a_form_eq2 + a_form_eq3)

    def time_march_primal(self,dt,pH_P,problem,input_n2,inputB_n2,inputB_n1):
        b_form_eq1 = (1/dt) * m_i(self.chi_1, pH_P.v_t) + 0.5 * eta_s(self.chi_1, pH_P.v_t, input_n2) \
                     + 0.5 * eta_p(self.chi_1, pH_P.p_t) + 0.5 * eta_k(self.chi_1, pH_P.w_t, self.kappa) \
                     + eta_B1(self.chi_1, inputB_n2, problem.n_ver, self.kappa)
        b_form_eq2 = 0.5 * eta_p_Tr(self.chi_0, pH_P.v_t) + eta_B2(self.chi_0, inputB_n1, problem.n_ver)
        b_form_eq3 = 0.0
        pH_P.time_march(b_form_eq1+b_form_eq2+b_form_eq3,dt,"gmres","amg")
        return pH_P.outputs()

    def time_march_dual(self,dt,pH_D,problem,input_2,inputB_1,inputB_0):
        b_form_eq1 = (1 / dt) * m_i(self.chiT_n1, pH_D.v_t) + 0.5 * etaT_s(self.chiT_n1, pH_D.v_t, input_2) \
                     + 0.5 * etaT_p(self.chiT_n1, pH_D.p_t) + 0.5 * etaT_k(self.chiT_n1, pH_D.w_t, self.kappa) \
                     + etaT_B1(self.chiT_n1, inputB_0, problem.n_ver)
        b_form_eq2 = 0.0
        b_form_eq3 = -0.5 * m_i(self.chiT_n2, pH_D.w_t) + 0.5 * etaT_k_Tr(self.chiT_n2, pH_D.v_t) \
                     + etaT_B2(self.chiT_n2, inputB_1, problem.n_ver)

        pH_D.time_march(b_form_eq1 + b_form_eq2 + b_form_eq3, dt,"gmres","amg")
        return pH_D.outputs()


    def solve(self, problem):
        # Get problem parameters
        mesh = problem.mesh
        dt, n_t, t_range = self.timestep(problem)
        self.kappa = problem.mu/problem.rho
        # print("Kinematic viscosity set to: ", self.kappa)

        # Define primal mixed elements
        ufl_cell = mesh.ufl_cell()
        P_1 = FiniteElement("N1curl", ufl_cell, self.pol_deg)
        P_0 = FiniteElement("CG", ufl_cell, self.pol_deg)
        if problem.dimM==3:
            P_2 = FiniteElement("RT", ufl_cell, self.pol_deg)
        elif problem.dimM==2:
            P_2 = FiniteElement("DG", ufl_cell, self.pol_deg-1)

        # Define dual mixed elements
        PT_n1 = FiniteElement("RT", ufl_cell, self.pol_deg)
        PT_n = FiniteElement("DG", ufl_cell, self.pol_deg - 1)
        if problem.dimM==3:
            PT_n2 = FiniteElement("N1curl", ufl_cell, self.pol_deg)
        elif problem.dimM==2:
            PT_n2 = FiniteElement("CG", ufl_cell, self.pol_deg)

        P_primal = MixedElement([P_1, P_2,P_0])
        P_dual = MixedElement([PT_n1,PT_n2,PT_n])

        # Define function spaces
        V_1 = FunctionSpace(mesh, P_1)
        V_2 = FunctionSpace(mesh, P_2)
        V_0 = FunctionSpace(mesh, P_0)
        V_primal = FunctionSpace(mesh, P_primal) # V_1 x V_2 x V_0
        VT_n1 = FunctionSpace(mesh, PT_n1)
        VT_n2 = FunctionSpace(mesh, PT_n2)
        VT_n = FunctionSpace(mesh, PT_n)
        V_dual = FunctionSpace(mesh, P_dual) # VT_n-1 x VT_n-2 x VT_n
        print("Function Space dimensions, Primal - Dual: ", [V_primal.dim(), V_dual.dim()])

        # Define test and trial functions
        self.init_test_trial_functions(V_primal, V_dual)

        # Define Function assigners
        fa_primal = FunctionAssigner(V_primal, [V_1, V_2, V_0])
        fa_dual = FunctionAssigner(V_dual, [VT_n1, VT_n2, VT_n])

        # Define Primal and Dual pH systems
        pH_primal = WeakPortHamiltonianSystemNS(V_primal, problem, "x_k")
        pH_dual = WeakPortHamiltonianSystemNS(V_dual, problem, "xT_kT")
        num_dof = np.sum(len(pH_primal.state_t_1.vector()) + len(pH_dual.state_t_1.vector()))
        # print("Num of DOFs: ", num_dof)

        # Set initial condition at t=0
        x_init = Function(V_primal, name="x initial")
        xT_init = Function(V_dual, name="xT initial")
        fa_primal.assign(x_init, problem.initial_conditions(V_1, V_2, V_0))
        fa_dual.assign(xT_init, problem.initial_conditions(VT_n1, VT_n2, VT_n))
        pH_primal.set_initial_condition(x_init)
        pH_dual.set_initial_condition(xT_init)

        # Set strong boundary conditions
        # primal system --> v_in
        bcv, bcw, bcp = problem.boundary_conditions(V_primal.sub(0), V_primal.sub(1), V_primal.sub(2), pH_primal.t_1)
        [pH_primal.set_boundary_condition(bc) for bc in bcv]
        # dual system --> w_in
        bcvT, bcwT, bcpT = problem.boundary_conditions(V_dual.sub(0), V_dual.sub(1), V_dual.sub(2), pH_dual.t_1)
        # [pH_dual.set_boundary_condition(bc) for bc in bcvT] # Does not converge
        # [pH_dual.set_boundary_condition(bc) for bc in bcwT]
        # TODO_Later: check correct implementation for multiple state inputs on boundary

        # Define Storage Arrays
        self.outputs_arr_primal = np.zeros((1 + n_t, 6))
        self.outputs_arr_dual = np.zeros((1 + n_t, 6))

        # Initial Functionals
        self.outputs_arr_primal[0] = pH_primal.outputs()
        # ||v_ex_t - v_t||,||w_ex_t - w_t||,||p_ex_t - p_t||,H_ex_t,H_t,||div(u_t)||
        self.outputs_arr_dual[0] = pH_dual.outputs()
        # ||v_ex_t - vT_t||,||w_ex_t - wT_t||,||p_ex_t - pT_t||,H_ex_t, HT_t, ||div(vT_t)||

        print("Initial outputs for primal system: ", self.outputs_arr_primal[0])
        print("Initial outputs for dual system: ", self.outputs_arr_dual[0])

        # Input for advancing primal system only
        vT_ex_tmid, wT_ex_tmid, pT_ex_tmid = problem.get_exact_sol_at_t(pH_primal.t_mid)
        input_n2 = interpolate(wT_ex_tmid,VT_n2)
        input_n1 = interpolate(vT_ex_tmid, VT_n1)

        # Input for advancing dual system only
        v_ex_tmid, w_ex_tmid, p_ex_tmid = problem.get_exact_sol_at_t(pH_dual.t_mid)
        input_2 = interpolate(w_ex_tmid, V_2)
        input_1 = interpolate(v_ex_tmid, V_1)
        input_0 = interpolate(p_ex_tmid, V_0)


        # Assemble LHS of Weak form (Single timestep)
        self.assemble_lhs_primal(dt, pH_primal,input_n2)
        self.assemble_lhs_dual(dt, pH_dual,input_2)

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ", self.pol_deg)
        if not (pH_primal.bcArr is None): print("Applying Strong Dirichlet B.C to Primal System")
        if not (pH_dual.bcArr is None): print("Applying Strong Dirichlet B.C to Dual System")
        print("==============")

        self.start_timing()

        # Advance dual system from t_0 --> t_1
        self.outputs_arr_dual[self._ts] = self.time_march_dual(dt, pH_dual, problem,input_2, input_1, input_0)
        print("Second output for dual system: ", self.outputs_arr_dual[self._ts])

        # Advance primal system from t_0 --> t_1
        self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, pH_primal, problem,input_n2,input_n2, input_n1 )
        print("Second output for primal system: ", self.outputs_arr_primal[self._ts])

        self.update(problem, dt)

        if self.stagger_time:
            # Time staggering strategy
            # 1. Average states of dual system at t_0 and t_1 to calculate t_1/2
            v_init, w_init, p_init = xT_init.split(deepcopy=True)
            pH_dual.v_t.vector()[:] += v_init.vector()[:]
            pH_dual.v_t.vector()[:] *= 0.5
            pH_dual.w_t.vector()[:] += w_init.vector()[:]
            pH_dual.w_t.vector()[:] *= 0.5
            pH_dual.p_t.vector()[:] += p_init.vector()[:]
            pH_dual.p_t.vector()[:] *= 0.5

            # 2. Reassign time variables of dual system to be at 1/2 time steps
            pH_dual.t.assign(dt / 2.0)
            pH_dual.t_1.assign((dt / 2.0) + dt)
            pH_dual.t_mid.assign(dt)

        # Time loop from t_1 onwards
        for t in tqdm(t_range[2:]):

            # Advance dual system from t_kT --> t_kT+1
            # Get in-domain vorticity input
            if self.couple_primal_dual:
                input_2 = pH_primal.w_t
            else:
                input_2 = interpolate(w_ex_tmid, V_2) # If exact solution exists
            # Get weak boundary inputs
            inputB_1 = interpolate(v_ex_tmid, V_1)
            inputB_0 = interpolate(p_ex_tmid, V_0)
            # Advance dual system
            self.outputs_arr_dual[self._ts] = self.time_march_dual(dt, pH_dual, problem, input_2, inputB_1, inputB_0)

            # Advance primal system from t_k --> t_k+1
            if self.couple_primal_dual:
                input_n2 = pH_dual.w_t
            else:
                input_n2 = interpolate(w_ex_tmid, VT_n2) # If exact solution exists

            # Advance primal system from t_ii --> t_ii+1
            inputB_n2 = interpolate(wT_ex_tmid, VT_n2)
            inputB_n1 = interpolate(vT_ex_tmid, VT_n1)
            self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, pH_primal, problem, input_n2, inputB_n2, inputB_n1)

            self.update(problem, dt)

# Generic mass form --> m(.)

def m_i(chi_i, alpha_i):
    form =  inner(chi_i,alpha_i) * dx
    return form

# Primal system weak forms --> eta(.)
# To be generalized to nD

def eta_s(chi_1, v_1, wT_1):
    form =  -dot(chi_1,cross(wT_1,v_1)) * dx
    return form

def eta_p(chi_1, p_0):
    form =  -dot(chi_1,grad(p_0)) * dx
    return form

def eta_k(chi_1, w_2, kappa):
    form = -dot(curl(chi_1),kappa*w_2) * dx
    return form

def eta_p_Tr(chi_0, v_1):
    form = dot(grad(chi_0),v_1) * dx
    return form

def eta_k_Tr(chi_2, v_1):
    form = dot(chi_2,curl(v_1)) * dx
    return form

def eta_B1(chi_1, wT_n2, n_vec, kappa):
    form = dot(cross(chi_1,kappa*wT_n2),n_vec) * ds
    return form

def eta_B2(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms --> eta^tilde(.)
# To be generalized to nD

def etaT_s(chi_2, vT_2, w_2):
    form = -dot(chi_2,cross(w_2,vT_2)) *dx
    return form

def etaT_p(chi_2,pT_3):
    form = dot(div(chi_2),pT_3)* dx
    return form

def etaT_k(chi_2,wT_1,kappa):
    form = -dot(chi_2,curl(kappa*wT_1))*dx
    return form

def etaT_p_Tr(chi_3, vT_2):
    form = dot(chi_3,div(vT_2)) * dx
    return form

def etaT_k_Tr(chi_1, vT_2):
    form = dot(curl(chi_1),vT_2) * dx
    return form

def etaT_B1(chi_2, p_0, n_vec):
    form = - p_0*dot(chi_2,n_vec) * ds
    return form

def etaT_B2(chi_1, v_1, n_vec):
    form = -dot(cross(chi_1,v_1),n_vec) * ds
    return form
