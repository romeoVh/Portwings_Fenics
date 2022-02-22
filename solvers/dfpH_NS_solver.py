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
        self.chiT_2 = self.chiT_1 = self.chiT_3 = None
        # Trial functions primal and dual
        self.v = self.w = self.p = None
        self.vT = self.wT = self.pT = None

    def __str__(self):
        return "DFPH_NS_3D"

    def init_test_trial_functions(self,V_primal,V_dual):
        chi_primal = TestFunction(V_primal)
        chi_dual = TestFunction(V_dual)
        self.chi_1 , self.chi_2 , self.chi_0  = split(chi_primal)
        self.chiT_2 , self.chiT_1 , self.chiT_3  = split(chi_dual)

        # Define Unknown Trial functions
        x_primal = TrialFunction(V_primal)
        x_dual = TrialFunction(V_dual)
        self.v , self.w , self.p  = split(x_primal)
        self.vT , self.wT , self.pT  = split(x_dual)

    def assemble_lhs_primal(self,dt,pH_P,input_1):
        a_form_eq1 = (1/dt)*m_1_form(self.chi_1,self.v)-0.5*eta_s_form(self.chi_1,self.v,input_1) \
                       -0.5*eta_p_form(self.chi_1,self.p) - 0.5*eta_k_form(self.chi_1,self.w,self.kappa)

        a_form_eq2 = -0.5*eta_p_Tr_form(self.chi_0,self.v)

        a_form_eq3 = 0.5*m_2_form(self.chi_2,self.w) - 0.5*eta_k_Tr_form(self.chi_2,self.v)

        # a_form_eq1 = (1 / dt) * m_1_form(self.chi_1, self.v) - 0.5 * eta_s_form(self.chi_1, self.v, input_1) \
        #                     -eta_p_form(self.chi_1,self.p) - 0.5*eta_k_form(self.chi_1,self.w,self.kappa)
        #
        # a_form_eq2 = eta_p_Tr_form(self.chi_0,self.v)
        #
        # a_form_eq3 = m_2_form(self.chi_2,self.w) - eta_k_Tr_form(self.chi_2,self.v)

        pH_P.A = assemble(a_form_eq1+a_form_eq2+a_form_eq3)

    def assemble_lhs_dual(self,dt,pH_dual):
        a_form_dual = None#m_form10(self.v_1, self.q_1, self.v_0, self.p_0) - 0.5 * dt * j_form10(self.v_1, self.q_1, self.v_0, self.p_0)
        pH_dual.A = assemble(a_form_dual)

    def time_march_primal(self,dt,pH_P,problem,input_1,input_2):
        b_form_eq1 = (1/dt)*m_1_form(self.chi_1,pH_P.v_t) + 0.5*eta_s_form(self.chi_1,pH_P.v_t,input_1)\
                            +0.5*eta_p_form(self.chi_1,pH_P.p_t)+ 0.5*eta_k_form(self.chi_1,pH_P.w_t,self.kappa)\
                            + eta_B1_form(self.chi_1,input_1,problem.n_ver,self.kappa)

        b_form_eq2 = 0.5*eta_p_Tr_form(self.chi_0,pH_P.v_t) + eta_B2_form(self.chi_0,input_2,problem.n_ver)

        b_form_eq3 = -0.5*m_2_form(self.chi_2,pH_P.w_t) + 0.5*eta_k_Tr_form(self.chi_2,pH_P.v_t)

        # b_form_eq1 = (1 / dt) * m_1_form(self.chi_1, pH_P.v_t) + 0.5 * eta_s_form(self.chi_1, pH_P.v_t, input_1) \
        #              + 0.5 * eta_k_form(self.chi_1, pH_P.w_t, self.kappa)
        #
        # b_form_eq2 = 0.0
        #
        # b_form_eq3 = 0.0

        b_32 = assemble(b_form_eq1+b_form_eq2+b_form_eq3)

        if not (pH_P.bcArr is None): [bc.apply(pH_P.A, b_32) for bc in pH_P.bcArr]
        solve(pH_P.A, pH_P.state_t_1.vector(), b_32, "gmres","hypre_amg")

        v_k_1, w_k_1, p_k_1 = pH_P.state_t_1.split(deepcopy=True)

        pH_P.v_t.assign(v_k_1)
        pH_P.w_t.assign(w_k_1)
        pH_P.p_t.assign(p_k_1)

        # Advance time step
        pH_P.advance_time(dt)
        return pH_P.outputs()

    def solve(self, problem):
        # Get problem parameters
        mesh = problem.mesh
        dt, n_t, t_range = self.timestep(problem)
        self.kappa = problem.mu/problem.rho
        # print("Kinematic viscosity set to: ", self.kappa)

        # Define mixed elements
        P_1 = FiniteElement("N1curl", tetrahedron, self.pol_deg)
        P_2 = FiniteElement("RT", tetrahedron, self.pol_deg)
        P_0 = FiniteElement("CG", tetrahedron, self.pol_deg)

        PT_2 = FiniteElement("RT", tetrahedron, self.pol_deg)
        PT_1 = FiniteElement("N1curl", tetrahedron, self.pol_deg)
        PT_3 = FiniteElement("DG", tetrahedron, self.pol_deg - 1)

        P_primal = MixedElement([P_1, P_2,P_0])
        P_dual = MixedElement([PT_2,PT_1,PT_3])

        # Define function spaces
        V_1 = FunctionSpace(mesh, P_1)
        V_2 = FunctionSpace(mesh, P_2)
        V_0 = FunctionSpace(mesh, P_0)
        V_primal = FunctionSpace(mesh, P_primal) # V_1 x V_2 x V_0
        VT_2 = FunctionSpace(mesh, PT_2)
        VT_1 = FunctionSpace(mesh, PT_1)
        VT_3 = FunctionSpace(mesh, PT_3)
        V_dual = FunctionSpace(mesh, P_dual) # VT_2 x VT_1 x VT_3
        print("Function Space dimensions, Primal - Dual: ", [V_primal.dim(), V_dual.dim()])

        # Define test and trial functions
        self.init_test_trial_functions(V_primal, V_dual)

        # Define Function assigners
        fa_primal = FunctionAssigner(V_primal, [V_1, V_2, V_0])
        fa_dual = FunctionAssigner(V_dual, [VT_2, VT_1, VT_3])

        # Define Primal and Dual pH systems
        pH_primal = WeakPortHamiltonianSystemNS(V_primal, problem, "x_k")
        pH_dual = WeakPortHamiltonianSystemNS(V_dual, problem, "xT_kT")
        num_dof = np.sum(len(pH_primal.state_t_1.vector()) + len(pH_dual.state_t_1.vector()))
        print("Num of DOFs: ", num_dof)

        # Set initial condition at t=0
        x_init = Function(V_primal, name="x initial")
        xT_init = Function(V_dual, name="xT initial")
        fa_primal.assign(x_init, problem.initial_conditions(V_1, V_2, V_0))
        fa_dual.assign(xT_init, problem.initial_conditions(VT_2, VT_1, VT_3))
        pH_primal.set_initial_condition(x_init)
        pH_dual.set_initial_condition(xT_init)

        # Set strong boundary conditions
        bcv, bcw, bcp = problem.boundary_conditions(V_primal.sub(0), V_primal.sub(1), V_primal.sub(2), pH_primal.t_1)
        [pH_primal.set_boundary_condition(bc) for bc in bcv]
        # TODO_Later: support multiple inputs for primal system

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
        v_ex_tmid, w_ex_tmid, p_ex_tmid = problem.get_exact_sol_at_t(pH_primal.t_mid)
        input_1 = interpolate(w_ex_tmid,VT_1)
        input_2 = interpolate(v_ex_tmid, VT_2)

        # Assemble LHS of Weak form (Single timestep)
        self.assemble_lhs_primal(dt, pH_primal,input_1)
        # self.assemble_lhs_dual(dt, pH_dual)

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ", self.pol_deg)
        if not (pH_primal.bcArr is None): print("Applying Strong Dirichlet B.C to Primal System")
        if not (pH_dual.bcArr is None): print("Applying Strong Dirichlet B.C to Dual System")
        print("==============")

        self.start_timing()

        # Advance dual system from t_0 --> t_1

        # Advance primal system from t_0 --> t_1
        self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, pH_primal, problem, input_1, input_2)
        print("Second output for primal system: ", self.outputs_arr_primal[self._ts])

        self.update(problem, dt)

        # Time loop from t_1 onwards
        for t in tqdm(t_range[2:]):
            # Advance 32 system from t_ii --> t_ii+1
            input_1 = interpolate(w_ex_tmid, VT_1)
            input_2 = interpolate(v_ex_tmid, VT_2)
            self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, pH_primal, problem, input_1, input_2)
            self.update(problem, dt)


def m_1_form(chi_1,v_1):
    form =  inner(chi_1,v_1) * dx
    return form

def m_2_form(chi_2,w_2):
    form =  inner(chi_2,w_2) * dx
    return form

def eta_s_form(chi_1,v_1,wT_1):
    form =  -dot(chi_1,cross(wT_1,v_1)) * dx
    return form

def eta_p_form(chi_1,p_0):
    form =  -dot(chi_1,grad(p_0)) * dx
    return form

def eta_k_form(chi_1,w_2,kappa):
    form = -dot(curl(chi_1),kappa*w_2) * dx
    return form

def eta_p_Tr_form(chi_0,v_1):
    form = dot(grad(chi_0),v_1) * dx
    return form

def eta_k_Tr_form(chi_2,v_1):
    form = dot(chi_2,curl(v_1)) * dx
    return form

def eta_B1_form(chi_1,wT_1,n_vec,kappa):
    form = 0# dot(cross(chi_1,kappa*wT_1),n_vec) * ds
    return form

def eta_B2_form(chi_0,vT_2,n_vec):
    form = 0#-chi_0*dot(vT_2,n_vec) * ds
    return form