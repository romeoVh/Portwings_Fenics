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
        self.v, self.w, self.p = split(x_primal)
        self.vT, self.wT, self.pT = split(x_dual)

    def assemble_lhs_primal(self,dt,pH_P,problem,input_n2):
        a_form_eq1 = (1/dt) * m_i(self.chi_1, self.v) - 0.5 * eta_s(problem.dimM,self.chi_1, self.v, input_n2) \
                     - eta_p(self.chi_1, self.p) - 0.5 * eta_k(problem.dimM,self.chi_1, self.w, self.kappa)
        a_form_eq2 = -eta_p_Tr(self.chi_0, self.v)
        a_form_eq3 = m_i(self.chi_2, self.w) - eta_k_Tr(problem.dimM,self.chi_2, self.v)
        pH_P.A = assemble(a_form_eq1+a_form_eq2+a_form_eq3)

    def assemble_lhs_dual(self,dt,pH_D,problem,input_2):
        a_form_eq1 = (1 / dt) * m_i(self.chiT_n1, self.vT) - 0.5 * etaT_s(problem.dimM,self.chiT_n1, self.vT, input_2) \
                     - etaT_p(self.chiT_n1, self.pT) - 0.5 * etaT_k(problem.dimM,self.chiT_n1, self.wT, self.kappa)
        a_form_eq2 = etaT_p_Tr(self.chiT_n, self.vT)
        a_form_eq3 = m_i(self.chiT_n2, self.wT) - etaT_k_Tr(problem.dimM,self.chiT_n2, self.vT)
        pH_D.A = assemble(a_form_eq1 + a_form_eq2 + a_form_eq3)

    def time_march_primal(self,dt,pH_P,problem,input_n2,inputB_n2,inputB_n1):
        b_form_eq1 = (1/dt) * m_i(self.chi_1, pH_P.v_t) + 0.5 * eta_s(problem.dimM,self.chi_1, pH_P.v_t, input_n2) \
                     + 0.5 * eta_k(problem.dimM,self.chi_1, pH_P.w_t, self.kappa) \
                     + self.bool_bcs_weak * eta_B1(problem.dimM,self.chi_1, inputB_n2, problem.n_ver, self.kappa)
        b_form_eq2 = self.bool_bcs_weak * eta_B2(self.chi_0, inputB_n1, problem.n_ver)
        b_form_eq3 = 0.0
        pH_P.time_march(b_form_eq1+b_form_eq2+b_form_eq3,dt,"gmres","amg")
        return pH_P.outputs(problem)

    def time_march_dual(self,dt,pH_D,problem,input_2,inputB_1,inputB_0):
        b_form_eq1 = (1 / dt) * m_i(self.chiT_n1, pH_D.v_t) + 0.5 * etaT_s(problem.dimM,self.chiT_n1, pH_D.v_t, input_2) \
                     + 0.5 * etaT_k(problem.dimM,self.chiT_n1, pH_D.w_t, self.kappa) \
                     + self.bool_bcs_weak * etaT_B1(self.chiT_n1, inputB_0, problem.n_ver)
        b_form_eq2 = 0.0
        b_form_eq3 = self.bool_bcs_weak * etaT_B2(problem.dimM,self.chiT_n2, inputB_1, problem.n_ver)

        pH_D.time_march(b_form_eq1 + b_form_eq2 + b_form_eq3, dt,"gmres","amg")
        return pH_D.outputs(problem)

    def explicit_step_primal(self, dt_0, problem, list_Vs):
        V_vel = list_Vs[0]
        V_vor = list_Vs[1]
        V_pr = list_Vs[2]
        list_init = problem.initial_conditions(V_vel, V_vor, V_pr)

        v_init = list_init[0]
        w_init = list_init[1]
        p_init = list_init[2]

        bc_vel, bc_vor, bc_pr = problem.boundary_conditions(V_vel, V_vor, V_pr, dt_0)

        chi_1 = TestFunction(V_vel)
        u_1 = TrialFunction(V_vel)

        a_form_vel = 1/dt_0 * m_i(chi_1, u_1)
        A_vel = assemble(a_form_vel)

        b_form_vel = (1/dt_0) * m_i(chi_1, v_init) + eta_s(problem.dimM, chi_1, v_init, w_init) \
                     + eta_p(chi_1, p_init) + eta_k(problem.dimM, chi_1, w_init, self.kappa) \
                     + self.bool_bcs_weak * eta_B1(problem.dimM, chi_1, w_init, problem.n_ver, self.kappa)
        b_vel = assemble(b_form_vel)

        v_sol = Function(V_vel)

        [bc.apply(A_vel, b_vel) for bc in bc_vel]
        solve(A_vel, v_sol.vector(), b_vel)

        chi_w = TestFunction(V_vor)
        w_trial = TrialFunction(V_vor)

        a_form_vor = m_i(chi_w, w_trial)
        A_vor = assemble(a_form_vor)

        b_form_vor = eta_k_Tr(problem.dimM, chi_w, v_sol)
        b_vor = assemble(b_form_vor)

        w_sol = Function(V_vor)

        [bc.apply(A_vor, b_vor) for bc in bc_vor]
        solve(A_vor, w_sol.vector(), b_vor)

        return v_sol, w_sol

    def solve(self, problem):
        # Get problem parameters
        self.bool_bcs_weak = 1
        if problem.periodic == True:
            self.bool_bcs_weak = 0
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
        self.pH_primal= WeakPortHamiltonianSystemNS(V_primal, problem, "x_k")
        self.pH_dual = WeakPortHamiltonianSystemNS(V_dual, problem, "xT_kT")
        num_dof = np.sum(len(self.pH_primal.state_t_1.vector()) + len(self.pH_dual.state_t_1.vector()))
        # print("Num of DOFs: ", num_dof)

        # Set initial condition at t=0
        x_init = Function(V_primal, name="x initial")
        xT_init = Function(V_dual, name="xT initial")
        fa_primal.assign(x_init, problem.initial_conditions(V_1, V_2, V_0))
        fa_dual.assign(xT_init, problem.initial_conditions(VT_n1, VT_n2, VT_n))
        self.pH_primal.set_initial_condition(x_init)
        self.pH_dual.set_initial_condition(xT_init)

        # Set strong boundary conditions
        # primal system --> v_in
        bcv, bcw, bcp = problem.boundary_conditions(V_primal.sub(0), V_primal.sub(1), V_primal.sub(2), self.pH_primal.t_1)
        [self.pH_primal.set_boundary_condition(bc) for bc in bcv]

        # dual system --> w_in
        bcvT, bcwT, bcpT = problem.boundary_conditions(V_dual.sub(0), V_dual.sub(1), V_dual.sub(2), self.pH_dual.t_1)
        #[self.pH_dual.set_boundary_condition(bc) for bc in bcvT] # Does not converge
        #[self.pH_dual.set_boundary_condition(bc) for bc in bcwT]
        # TODO_Later: check correct implementation for multiple state inputs on boundary

        # Initialize problem outputs
        self.pH_primal.prob_output_arr = problem.init_outputs(self.pH_primal.t)
        self.pH_dual.prob_output_arr = problem.init_outputs(self.pH_dual.t)
        num_prob_outputs = len(self.pH_primal.prob_output_arr)

        # Define Storage Arrays
        num_outputs = self.pH_primal.init_outputs()
        num_outputs = self.pH_dual.init_outputs()

        self.outputs_arr_primal = np.zeros((1 + n_t, num_prob_outputs+num_outputs))
        self.outputs_arr_dual = np.zeros((1 + n_t, num_prob_outputs+num_outputs))

        # Initial Functionals
        self.outputs_arr_primal[0] = self.pH_primal.outputs(problem)
        # Problem specific outpus + H_t,||div(v_t)||
        self.outputs_arr_dual[0] = self.pH_dual.outputs(problem)
        # Problem specific outpus + HT_t, ||div(vT_t)||

        print("Initial outputs for primal system: ", self.outputs_arr_primal[0])
        print("Initial outputs for dual system: ", self.outputs_arr_dual[0])

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ", self.pol_deg)
        if not (self.pH_primal.bcArr is None): print("Applying Strong Dirichlet B.C to Primal System")
        if not (self.pH_dual.bcArr is None): print("Applying Strong Dirichlet B.C to Dual System")
        print("==============")

        self.start_timing()

        # ------------------------------------------
        # Initial time advancement from t_0 to t_1
        # ------------------------------------------
        # Euler step for the primal system
        v_half, w_half = self.explicit_step_primal(dt/2, problem, [V_1, V_2, V_0])

        self.pH_primal.v_t.assign(v_half)
        self.pH_primal.w_t.assign(w_half)
        self.pH_primal.p_t.assign(problem.initial_conditions(V_1, V_2, V_0)[2])
        self.pH_primal.advance_time(dt/2)
        # Update outputs of the problems

        self.outputs_arr_primal[self._ts] = self.pH_primal.outputs(problem)

        # ------------------------------------------
        # End of inital time advance
        # ------------------------------------------
        self.update(problem, dt/2)

        # Time loop from t_1 onwards
        for t in tqdm(t_range[1:-1]):

            # Advance dual system from t_kT --> t_kT+1
            self.assemble_lhs_dual(dt, self.pH_dual, problem, self.pH_primal.w_t)
            # Get weak boundary inputs
            if self.couple_primal_dual:
                input_1 = self.pH_primal.v_t
                input_0 = self.pH_primal.p_t
            else:
                input_1 = interpolate(v_ex_tmid, V_1)
                input_0 = interpolate(p_ex_tmid, V_0)
                # Should be changed to get them from problem if False

            self.outputs_arr_dual[self._ts] = self.time_march_dual(dt, self.pH_dual, problem, self.pH_primal.w_t,
                                                                   input_1, input_0)

            # Advance primal system from t_k --> t_k+1
            self.assemble_lhs_primal(dt, self.pH_primal, problem, self.pH_dual.w_t)
            # Get weak boundary inputs
            if self.couple_primal_dual:
                input_n2 = self.pH_dual.w_t
                input_n1 = self.pH_dual.v_t
            else:
                input_n2 = interpolate(wT_ex_tmid,VT_n2)
                input_n1 = interpolate(vT_ex_tmid, VT_n1)
                # Should be changed to get them from problem if False

            self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, self.pH_primal, problem,\
                                                                       self.pH_dual.w_t, input_n2,input_n1)

            self.update(problem, dt)

# Generic mass form --> m(.)

def m_i(chi_i, alpha_i):
    form = inner(chi_i,alpha_i) * dx
    return form

# Primal system weak forms --> eta(.)

def eta_s(dimM,chi_1, v_1, wT_n2):
    if dimM==3:
        form = inner(chi_1,cross(v_1, wT_n2)) * dx
    elif dimM==2:
        form = dot(wT_n2, v_1[1]*chi_1[0] - v_1[0]*chi_1[1]) * dx
    return form

def eta_p(chi_1, p_0):
    form = -inner(chi_1,grad(p_0)) * dx
    return form

def eta_k(dimM,chi_1, w_2, kappa):
    if dimM==3:
        form = -inner(curl(chi_1),kappa*w_2) * dx
    elif dimM==2:
        form = -dot(curl2D(chi_1),kappa*w_2) * dx
    return form

def eta_p_Tr(chi_0, v_1):
    form = -inner(grad(chi_0),v_1) * dx
    return form

def eta_k_Tr(dimM,chi_2, v_1):
    if dimM==3:
        form = inner(chi_2,curl(v_1)) * dx
    elif dimM==2:
        form = dot(chi_2,curl2D(v_1)) * dx
    return form

def eta_B1(dimM, chi_1, wT_n2, n_vec, kappa):
    if dimM==3:
        form = dot(cross(chi_1,kappa*wT_n2),n_vec) * ds
    elif dimM==2:
        form = kappa*wT_n2*dot(as_vector((chi_1[1], -chi_1[0])), n_vec) * ds
    return form

def eta_B2(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms --> eta^tilde(.)
# To be generalized to nD

def etaT_s(dimM,chi_2, vT_2, w_2):
    if dimM==3:
        form = inner(chi_2,cross(vT_2, w_2)) * dx
    elif dimM==2:
        form = dot(w_2, vT_2[1]*chi_2[0] - vT_2[0]*chi_2[1]) * dx
    return form

def etaT_p(chi_2,pT_3):
    form = inner(div(chi_2),pT_3) * dx
    return form

def etaT_k(dimM,chi_2,wT_1,kappa):
    if dimM == 3:
        form = -inner(chi_2, kappa*curl(wT_1)) * dx
    elif dimM == 2:
        form = -dot(chi_2, kappa*rot2D(wT_1)) * dx
        # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    return form

def etaT_p_Tr(chi_3, vT_2):
    form = inner(chi_3, div(vT_2)) * dx
    return form

def etaT_k_Tr(dimM,chi_1, vT_2):
    if dimM == 3:
        form = inner(curl(chi_1), vT_2) * dx
    elif dimM == 2:
        form = dot(rot2D(chi_1), vT_2) * dx
    return form

def etaT_B1(chi_2, p_0, n_vec):
    form = -p_0*dot(chi_2,n_vec) * ds
    return form

def etaT_B2(dimM,chi_1, v_1, n_vec):
    if dimM == 3:
        form = -dot(cross(chi_1, v_1), n_vec) * ds
    elif dimM == 2:
        form = chi_1*dot(as_vector((v_1[1], -v_1[0])), n_vec) * ds
    return form

def curl2D(v):
    return v[1].dx(0) - v[0].dx(1)

def rot2D(w):
    return as_vector((w.dx(1), -w.dx(0)))
