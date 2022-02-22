from fenics import *
from time import time
from tqdm import tqdm
from math_expressions import *
from solver_base import *
from weak_pH_system import *

class DualFieldPHWaveSolver(SolverBase):
    # "Dual Field port-Hamiltonian scheme for 3D Wave equation."
    # Could be generalized to any linear canonical pH system

    def __init__(self, options):
        SolverBase.__init__(self, options)
        self.stagger_time = options["stagger_time"]
        self.bnd_cond = options["bnd_cnd"]
        self.couple_primal_dual = options["couple_primal_dual"]

        # Test functions primal and dual
        self.v_3 = self.v_2 = self.v_0 = self.v_1 = None
        # Trial functions primal and dual
        self.p_3 = self.q_2 = self.p_0 = self.q_1 = None

    def init_test_trial_functions(self,V_32,V_01):
        v_32 = TestFunction(V_32)
        v_01 = TestFunction(V_01)
        self.v_3, self.v_2 = split(v_32)
        self.v_0, self.v_1 = split(v_01)

        # Define Unknown Trial functions
        e_32 = TrialFunction(V_32)
        e_01 = TrialFunction(V_01)
        self.p_3, self.q_2 = split(e_32)
        self.p_0, self.q_1 = split(e_01)

    def __str__(self):
        return "DFPHWave3D"

    def assemble_lhs_primal(self,dt,pH_32):
        a_form32 = m_form32(self.v_3, self.p_3, self.v_2, self.q_2) - 0.5 * dt * j_form32(self.v_3, self.p_3, self.v_2, self.q_2)
        pH_32.A = assemble(a_form32)


    def assemble_lhs_dual(self,dt,pH_01):
        a_form01 = m_form10(self.v_1, self.q_1, self.v_0, self.p_0) - 0.5 * dt * j_form10(self.v_1, self.q_1, self.v_0, self.p_0)
        pH_01.A = assemble(a_form01)

    def time_march_primal(self,dt,pH_32,problem,input_0):
        b_form32 = m_form32(self.v_3, pH_32.p_t, self.v_2, pH_32.q_t) \
                   + dt * (0.5 * j_form32(self.v_3, pH_32.p_t, self.v_2, pH_32.q_t)+ bdflow32(self.v_2, input_0,problem.n_ver))
        b_32 = assemble(b_form32)
        if not (pH_32.bcArr is None): [bc.apply(pH_32.A, b_32) for bc in pH_32.bcArr]
        solve(pH_32.A, pH_32.state_t_1.vector(), b_32)

        p_3_k_1, q_2_k_1 = pH_32.state_t_1.split(deepcopy=True)
        pH_32.p_t.assign(p_3_k_1)
        pH_32.q_t.assign(q_2_k_1)

        # Advance time step
        pH_32.advance_time(dt)
        return pH_32.outputs()

    def time_march_dual(self,dt,pH_01,problem,input_2):
        b_form10 = m_form10(self.v_1, pH_01.q_t, self.v_0, pH_01.p_t) + dt * (
                    0.5 * j_form10(self.v_1, pH_01.q_t, self.v_0, pH_01.p_t) + bdflow10(self.v_0, input_2, problem.n_ver))
        b_10 = assemble(b_form10)
        if not (pH_01.bcArr is None): [bc.apply(pH_01.A, b_10) for bc in pH_01.bcArr]

        solve(pH_01.A, pH_01.state_t_1.vector(), b_10)

        p_0_kT_1, q_1_kT_1 = pH_01.state_t_1.split(deepcopy=True)
        pH_01.p_t.assign(p_0_kT_1)
        pH_01.q_t.assign(q_1_kT_1)

        # Advance time step
        pH_01.advance_time(dt)
        return pH_01.outputs()

    def end_of_loop(self, problem, t_range, dt, outputs_32, outputs_01):
        if self.stagger_time:
            t_range_shifted = np.roll(t_range, 1) +dt/2.0
            t_range_shifted[0] = 0.0
        else:
            t_range_shifted = t_range
        #print(t_range)
        #print(t_range_shifted)
        plt.subplot(1,2,1)
        plt.bar(t_range, np.sum(outputs_32[:,0:2], axis=1), width=float(dt) / 2)
        plt.title("L2 error of p and q for primal system")
        plt.subplot(1, 2, 2)
        plt.bar(t_range_shifted, np.sum(outputs_01[:, 0:2], axis=1), width=float(dt) / 2)
        plt.title("L2 error of p and q for dual system")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(t_range, outputs_32[:,2:4])
        plt.legend(['H_32', 'H_ex'])
        plt.subplot(1, 2, 2)
        plt.plot(t_range_shifted, outputs_01[:, 2:4])
        plt.legend(['H_01', 'H_ex'])
        plt.show()

    def solve(self, problem):
        # Get problem parameters
        mesh = problem.mesh
        dt, n_t, t_range = self.timestep(problem)

        # Define mixed elements
        P_0 = FiniteElement("CG", tetrahedron, self.pol_deg)
        P_1 = FiniteElement("N1curl", tetrahedron, self.pol_deg)
        P_2 = FiniteElement("RT", tetrahedron, self.pol_deg)
        P_3 = FiniteElement("DG", tetrahedron, self.pol_deg - 1)

        P_32 = MixedElement([P_3, P_2])
        P_01 = MixedElement([P_0,P_1])

        # Define function spaces
        V_3 = FunctionSpace(mesh, P_3)
        V_2 = FunctionSpace(mesh, P_2)
        V_1 = FunctionSpace(mesh, P_1)
        V_0 = FunctionSpace(mesh, P_0)
        V_32 = FunctionSpace(mesh, P_32)
        V_01 = FunctionSpace(mesh, P_01)
        print("Function Space dimensions: ", [[V_3.dim(), V_2.dim()], [V_1.dim(), V_0.dim()]])

        # Define test and trial functions
        self.init_test_trial_functions(V_32,V_01)

        # Define Function assigners
        fa_32 = FunctionAssigner(V_32, [V_3, V_2])
        fa_01 = FunctionAssigner(V_01, [V_0, V_1])

        # Define Primal and Dual pH systems
        pH_32 = WeakPortHamiltonianSystem(V_32,problem,"e_32_k")
        pH_01 = WeakPortHamiltonianSystem(V_01,problem,"e_01_kT")
        #print(pH_32.state_t,pH_32.state_t_1,pH_01.state_t,pH_01.state_t_1)

        num_dof = np.sum(len(pH_32.state_t_1.vector()) + len(pH_01.state_t_1.vector()))
        #print(num_dof)

        # Set initial condition at t=0
        e_32_init = Function(V_32, name="e_32 initial")
        e_01_init = Function(V_01, name="e_01 initial")
        fa_32.assign(e_32_init, problem.initial_conditions(V_3,V_2))
        fa_01.assign(e_01_init, problem.initial_conditions(V_0,V_1))
        pH_32.set_initial_condition(e_32_init)
        pH_01.set_initial_condition(e_01_init)

        # Set strong boundary conditions
        if self.bnd_cond == "N":
            pH_32.set_boundary_condition(problem,1)
        elif self.bnd_cond == "D":
            pH_01.set_boundary_condition(problem,0)



        # Define Storage Arrays
        outputs_32 = np.zeros((1 + n_t, 4))
        outputs_01 = np.zeros((1 + n_t, 4))

        # Initial Functionals
        outputs_32[0] = pH_32.outputs() # ||p_ex - p_3_t||,||q_ex - q_2_t||, H_32_t, H_ex_t
        outputs_01[0] = pH_01.outputs() # ||p_ex - p_0_t||,||q_ex - q_1_t||, H_01_t, H_ex_t

        print("Initial outputs for 32 system: ", outputs_32[0])
        print("Initial outputs for 01 system: ", outputs_01[0])

        # Define Test and Trial functions
        self.init_test_trial_functions(V_32,V_01)

        # Assemble LHS of Weak form (Single timestep)
        self.assemble_lhs_primal(dt,pH_32)
        self.assemble_lhs_dual(dt, pH_01)

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ",self.pol_deg)
        if not (pH_32.bcArr is None): print("Applying Strong Dirichlet B.C to 32 System")
        if not (pH_01.bcArr is None): print("Applying Strong Dirichlet B.C to 01 System")
        print("==============")

        self.start_timing()

        # Advance 01 system from t_0 --> t_1
        p_ex_mid_01, q_ex_mid_01 = problem.get_exact_sol_at_t(pH_01.t_mid)
        input_2 = -interpolate(q_ex_mid_01, V_2)
        outputs_01[self._ts] = self.time_march_dual(dt,pH_01,problem,input_2)
        # print("Second outputs for 01 system: ", outputs_01[1])

        # Advance 32 system from t_0 --> t_1
        p_ex_mid_32, q_ex_mid_32 = problem.get_exact_sol_at_t(pH_32.t_mid)
        input_0 = interpolate(p_ex_mid_32, V_0)
        outputs_32[self._ts] = self.time_march_primal(dt,pH_32,problem,input_0)
        #print("Second outputs for 32 system: ", outputs_32)

        self.update(problem,dt)

        if self.stagger_time:
            # Time staggering strategy
            # 1. Average states of 01 system at t_0 and t_1 to calculate t_1/2
            p_0_init, q_1_init = e_01_init.split(deepcopy=True)
            pH_01.p_t.vector()[:] += p_0_init.vector()[:]
            pH_01.p_t.vector()[:] *= 0.5
            pH_01.q_t.vector()[:] += q_1_init.vector()[:]
            pH_01.q_t.vector()[:] *= 0.5

            # Alternative code that didn't work
            # pH_01.p_t.assign(0.5*(pH_01.p_t + p_0_init))
            # pH_01.q_t.assign(0.5*(pH_01.q_t + q_1_init))

            # 2. Reassign time variables of 01 to be at 1/2 time steps
            pH_01.t.assign(dt / 2.0)
            pH_01.t_1.assign((dt / 2.0) + dt)
            pH_01.t_mid.assign(dt)

        # Time loop from t_1 onwards
        for t in tqdm(t_range[2:]):

            # Advance 01 system from t_ii --> t_ii+1
            if self.couple_primal_dual:
                input_2 = - pH_32.q_t
            else:
                input_2 = -interpolate(q_ex_mid_01, V_2)

            outputs_01[self._ts] = self.time_march_dual(dt, pH_01, problem, input_2)

            # Advance 32 system from t_ii --> t_ii+1
            if self.couple_primal_dual:
                input_0 = pH_01.p_t
            else:
                input_0 = interpolate(p_ex_mid_32, V_0)

            outputs_32[self._ts] = self.time_march_primal(dt, pH_32, problem, input_0)

            self.update(problem,t)

        self.end_of_loop(problem, t_range, dt, outputs_32, outputs_01)



def m_form32(v_3, p_3, v_2, q_2):
        m_form = inner(v_3, p_3) * dx + inner(v_2, q_2) * dx
        return m_form

def m_form10(v_1, q_1, v_0, p_0):
        m_form = inner(v_1, q_1) * dx + inner(v_0, p_0) * dx
        return m_form

def j_form32(v_3, p_3, v_2, q_2):
        j_form = -dot(v_3, div(q_2)) * dx + dot(div(v_2), p_3) * dx
        return j_form

def j_form10(v_1, q_1, v_0, p_0):
        j_form = -dot(v_1, grad(p_0)) * dx + dot(grad(v_0), q_1) * dx
        return j_form

def bdflow32(v_2, u_0,n_vec):
        b_form = -dot(v_2, n_vec) * u_0 * ds
        return b_form

def bdflow10(v_0, u_2,n_vec):
        b_form = v_0 * dot(u_2, n_vec) * ds
        return b_form