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
        self.v , self.w , self.p  = split(x_primal)
        self.vT , self.wT , self.pT  = split(x_dual)

    def assemble_lhs_primal(self,dt,pH_P,problem,input_n2):
        a_form_eq1 = (1/dt) * m_i(self.chi_1, self.v) - 0.5 * eta_s(problem.dimM,self.chi_1, self.v, input_n2) \
                     - 0.5 * eta_p(self.chi_1, self.p) - 0.5 * eta_k(problem.dimM,self.chi_1, self.w, self.kappa)
        a_form_eq2 = -0.5 * eta_p_Tr(problem.dimM,self.chi_0, self.v)
        a_form_eq3 = m_i(self.chi_2, self.w) - eta_k_Tr(problem.dimM,self.chi_2, self.v)
        pH_P.A = assemble(a_form_eq1+a_form_eq2+a_form_eq3)

    def assemble_lhs_dual(self,dt,pH_D,problem,input_2):
        a_form_eq1 = (1 / dt) * m_i(self.chiT_n1, self.vT) - 0.5 * etaT_s(problem.dimM,self.chiT_n1, self.vT, input_2) \
                     - 0.5 * etaT_p(problem.dimM,self.chiT_n1, self.pT) - 0.5 * etaT_k(problem.dimM,self.chiT_n1, self.wT, self.kappa)
        a_form_eq2 = etaT_p_Tr(self.chiT_n, self.vT)
        a_form_eq3 = 0.5 * m_i(self.chiT_n2, self.wT) - 0.5 * etaT_k_Tr(problem.dimM,self.chiT_n2, self.vT)
        pH_D.A = assemble(a_form_eq1 + a_form_eq2 + a_form_eq3)

    def time_march_primal(self,dt,pH_P,problem,input_n2,inputB_n2,inputB_n1):
        b_form_eq1 = (1/dt) * m_i(self.chi_1, pH_P.v_t) + 0.5 * eta_s(problem.dimM,self.chi_1, pH_P.v_t, input_n2) \
                     + 0.5 * eta_p(self.chi_1, pH_P.p_t) + 0.5 * eta_k(problem.dimM,self.chi_1, pH_P.w_t, self.kappa) \
                     + self.bool_bcs_weak * eta_B1(problem.dimM,self.chi_1, inputB_n2, problem.n_ver, self.kappa)
        b_form_eq2 = 0.5 * eta_p_Tr(problem.dimM,self.chi_0, pH_P.v_t) + self.bool_bcs_weak * eta_B2(self.chi_0, inputB_n1, problem.n_ver)
        b_form_eq3 = 0.0
        pH_P.time_march(b_form_eq1+b_form_eq2+b_form_eq3,dt,"gmres","amg")
        return pH_P.outputs(problem)

    def time_march_dual(self,dt,pH_D,problem,input_2,inputB_1,inputB_0):
        b_form_eq1 = (1 / dt) * m_i(self.chiT_n1, pH_D.v_t) + 0.5 * etaT_s(problem.dimM,self.chiT_n1, pH_D.v_t, input_2) \
                     + 0.5 * etaT_p(problem.dimM,self.chiT_n1, pH_D.p_t) + 0.5 * etaT_k(problem.dimM,self.chiT_n1, pH_D.w_t, self.kappa) \
                     + self.bool_bcs_weak * etaT_B1(problem.dimM,self.chiT_n1, inputB_0, problem.n_ver)
        b_form_eq2 = 0.0
        b_form_eq3 = -0.5 * m_i(self.chiT_n2, pH_D.w_t) + 0.5 * etaT_k_Tr(problem.dimM,self.chiT_n2, pH_D.v_t) \
                     + self.bool_bcs_weak * etaT_B2(problem.dimM,self.chiT_n2, inputB_1, problem.n_ver)

        pH_D.time_march(b_form_eq1 + b_form_eq2 + b_form_eq3, dt,"gmres","amg")
        return pH_D.outputs(problem)


    def solve(self, problem):
        # Get problem parameters
        self.bool_bcs_weak = 1
        if problem.__module__.split(".")[-1].lower() == "TaylorGreen":
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
        self.pH_primal.prob_output_arr =  problem.init_outputs(self.pH_primal.t)
        self.pH_dual.prob_output_arr =  problem.init_outputs(self.pH_dual.t)
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
        # Current Solution valid if exact solution exists and would be replaced by solving full nonlinear system
        # 1. Get primal & dual system states at t_1/2
        v_ex_tmid, w_ex_tmid, p_ex_tmid = problem.get_exact_sol_at_t(self.pH_dual.t_mid)
        vT_ex_tmid, wT_ex_tmid, pT_ex_tmid = problem.get_exact_sol_at_t(self.pH_primal.t_mid)

        # 2. Advance primal system
        input_n2 = interpolate(wT_ex_tmid,VT_n2)
        input_n1 = interpolate(vT_ex_tmid, VT_n1)
        self.assemble_lhs_primal(dt, self.pH_primal, problem, input_n2)
        self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, self.pH_primal, problem, input_n2, input_n2,
                                                                   input_n1)
        print("Second output for primal system: ", self.outputs_arr_primal[self._ts])
        # 3. Average states of primal system at t_0 and t_1 to calculate t_1/2
        v_tmid, w_tmid, p_tmid = x_init.split(deepcopy=True)
        v_tmid.vector()[:] += self.pH_primal.v_t.vector()[:]
        v_tmid.vector()[:] *= 0.5
        w_tmid.vector()[:] += self.pH_primal.w_t.vector()[:]
        w_tmid.vector()[:] *= 0.5
        p_tmid.vector()[:] += self.pH_primal.p_t.vector()[:]
        p_tmid.vector()[:] *= 0.5
        # 4. Advance dual system
        self.assemble_lhs_dual(dt, self.pH_dual, problem, w_tmid)
        self.outputs_arr_dual[self._ts] = self.time_march_dual(dt, self.pH_dual, problem,w_tmid, v_tmid, p_tmid)
        print("Second output for dual system: ", self.outputs_arr_dual[self._ts])

        # 5. Average states of dual system at t_0 and t_1 to calculate t_1/2
        vT_init, wT_init, pT_init = xT_init.split(deepcopy=True)
        self.pH_dual.v_t.vector()[:] += vT_init.vector()[:]
        self.pH_dual.v_t.vector()[:] *= 0.5
        self.pH_dual.w_t.vector()[:] += wT_init.vector()[:]
        self.pH_dual.w_t.vector()[:] *= 0.5
        self.pH_dual.p_t.vector()[:] += pT_init.vector()[:]
        self.pH_dual.p_t.vector()[:] *= 0.5

        # 6. Reassign time variables of dual system to be at 1/2 time steps
        self.pH_dual.t.assign(dt / 2.0)
        self.pH_dual.t_1.assign((dt / 2.0) + dt)
        self.pH_dual.t_mid.assign(dt)

        # ------------------------------------------
        # End of inital time advance
        # ------------------------------------------
        self.update(problem, dt)

        # Time loop from t_1 onwards
        for t in tqdm(t_range[2:]):
            # Question: Do we need to always reassemble the LHS ??

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

            self.outputs_arr_primal[self._ts] = self.time_march_primal(dt, self.pH_primal, problem, self.pH_dual.w_t, input_n2,input_n1)

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

            self.outputs_arr_dual[self._ts] = self.time_march_dual(dt, self.pH_dual, problem, self.pH_primal.w_t, input_1, input_0)


            self.update(problem, dt)

# Generic mass form --> m(.)

def m_i(chi_i, alpha_i):
    form =  inner(chi_i,alpha_i) * dx
    return form

# Primal system weak forms --> eta(.)

def eta_s(dimM,chi_1, v_1, wT_n2):
    if dimM==3:
        form =  -inner(chi_1,cross(wT_n2,v_1)) * dx
    elif dimM==2:
        form = -dot(wT_n2, v_1[0]*chi_1[1] - v_1[1]*chi_1[0]) * dx
    return form

def eta_p(chi_1, p_0):
    form =  -inner(chi_1,grad(p_0)) * dx
    return form

def eta_k(dimM,chi_1, w_2, kappa):
    if dimM==3:
        form = -inner(curl(chi_1),kappa*w_2) * dx
    elif dimM==2:
        form = -dot(chi_1[0].dx(1)-chi_1[1].dx(0),kappa*w_2) * dx# chi_1.dx(0)
    return form

def eta_p_Tr(dimM,chi_0, v_1):
    form = pow(-1,dimM-1)*inner(grad(chi_0),v_1) * dx
    return form

def eta_k_Tr(dimM,chi_2, v_1):
    if dimM==3:
        form = inner(chi_2,curl(v_1)) * dx
    elif dimM==2:
        form = dot(chi_2,v_1[0].dx(1)-v_1[1].dx(0)) * dx
    return form

def eta_B1(dimM, chi_1, wT_n2, n_vec, kappa):
    if dimM==3:
        form = dot(cross(chi_1,kappa*wT_n2),n_vec) * ds
    elif dimM==2:
        form = 0.0 # How to do ?
    return form

def eta_B2(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms --> eta^tilde(.)
# To be generalized to nD

def etaT_s(dimM,chi_2, vT_2, w_2):
    if dimM==3:
        form =  -inner(chi_2,cross(w_2,vT_2)) *dx
    elif dimM==2:
        form = -dot(w_2, vT_2[0]*chi_2[1] - vT_2[1]*chi_2[0]) * dx
    return form

def etaT_p(dimM,chi_2,pT_3):
    form = pow(-1,dimM-1)*inner(div(chi_2),pT_3)* dx
    return form

def etaT_k(dimM,chi_2,wT_1,kappa):
    if dimM == 3:
        form = -inner(chi_2, curl(kappa * wT_1)) * dx
    elif dimM == 2:
        form = dot(chi_2, kappa*as_vector((wT_1.dx(1),-wT_1.dx(0)))) * dx
        # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    return form

def etaT_p_Tr(chi_3, vT_2):
    form = inner(chi_3,div(vT_2)) * dx
    return form

def etaT_k_Tr(dimM,chi_1, vT_2):
    if dimM == 3:
        form = inner(curl(chi_1),vT_2) * dx
    elif dimM == 2:
        form = -dot(as_vector((chi_1.dx(1),-chi_1.dx(0))),vT_2) * dx
    return form

def etaT_B1(dimM,chi_2, p_0, n_vec):
    form = pow(-1,dimM)*p_0*dot(chi_2,n_vec) * ds
    return form

def etaT_B2(dimM,chi_1, v_1, n_vec):
    if dimM == 3:
        form = -dot(cross(chi_1, v_1), n_vec) * ds
    elif dimM == 2:
        form = 0.0  # How ?
    return form

