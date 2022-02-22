from fenics import *
from time import time
from tqdm import tqdm
from solvers.solver_base import *

class IPCS_Solver(SolverBase):
    "Incremental pressure-correction scheme."

    def __init__(self, options):
        SolverBase.__init__(self, options)
        # Test functions
        self.v = self.q = None
        # Trial functions
        self.u = self.p = None

    def init_test_trial_functions(self,V,Q):
        # Test functions for u and p
        self.v = TestFunction(V)
        self.q = TestFunction(Q)
        # Unknown Trial functions for u and p
        self.u = TrialFunction(V)
        self.p = TrialFunction(Q)


    def __str__(self):
        return "IPCS"

    def calculate_outputs(self):
        # Calculates ||u_ex_t - u_t||,||p_ex_t - p_t||, H_t, H_ex_t, ||div(u_t)||, ||div(u_ex_t)||
        err_u = errornorm(self.u_ex_t, self.u_t, norm_type="L2")
        err_p = errornorm(self.p_ex_t, self.p_t, norm_type="L2")
        H_ex_t = assemble(self.H_ex_t)
        H_t = assemble(self.H_t)
        divU = assemble(self.div_u_t)
        #divUex = assemble(self.div_u_ex_t)
        return np.array([err_u, err_p, H_t, H_ex_t,divU])

    def weak_form(self,dt,problem):

        mu = Constant(problem.mu)
        rho = Constant(problem.rho)
        k = Constant(dt)
        n_vec = FacetNormal(problem.n_ver)

        # Variational problem for step 1 : calculate Tentative velocity
        U = 0.5 * (self.u_t + self.u)
        F1 = rho * dot((self.u - self.u_t) / k, self.v) * dx + rho * dot(dot(self.u_t, nabla_grad(self.u_t)), self.v) * dx \
             + inner(sigma(U, self.p_t,mu), epsilon(self.v)) * dx \
             + dot(self.p_t * n_vec, self.v) * ds - dot(mu * nabla_grad(U) * n_vec, self.v) * ds
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Variational problem for step 2: correct pressure
        self.a2 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx
        self.L2 = dot(nabla_grad(self.p_t), nabla_grad(self.q)) * dx - (1 / k) * div(self.u_t_1) * self.q * dx

        # Variational problem for step 3: correct velocity
        self.a3 = dot(self.u, self.v) * dx
        self.L3 = dot(self.u_t_1, self.v) * dx - k * dot(nabla_grad(self.p_t_1 - self.p_t), self.v) * dx

    def solve(self, problem):
        # Get problem parameters
        mesh = problem.mesh
        dt, n_t, t_range = self.timestep(problem)

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", self.pol_deg)
        Q = FunctionSpace(mesh, "CG", self.pol_deg-1)
        print("Function Space dimensions: ", [V.dim(), Q.dim()])

        # Define test and trial functions
        self.init_test_trial_functions(V,Q)

        # Define time variables
        t_c = Constant(0.0)  # Current time step
        t_1_c = Constant(problem.dt)  # Next time step

        # Define state variables at t and t+1
        self.u_t = Function(V,name="u t")
        self.p_t = Function(Q,name="p t")
        self.u_t_1 = Function(V, name="u t+1")
        self.p_t_1 = Function(Q, name="p t+1")
        num_dof = np.sum(len(self.u_t_1.vector()) + len(self.p_t_1.vector()))
        #print(num_dof)


        # Set initial condition at t=0
        u_init, w_init, p_init = problem.initial_conditions(V,V,Q) # In 3D, omega is assumed to be in V
        self.u_t.assign(u_init)
        self.p_t.assign(p_init)
        # Energy of System and divergence
        self.H_t = 0.5 * (inner(self.u_t, self.u_t) * dx)
        self.div_u_t = div(self.u_t)* dx

        # Exact State and Energy
        self.u_ex_t, self.w_ex_t, self.p_ex_t = problem.get_exact_sol_at_t(t_c)
        self.H_ex_t = 0.5 * (inner(self.u_ex_t, self.u_ex_t) * dx(domain=problem.mesh))
        #self.div_u_ex_t = div(self.u_ex_t)

        # # Set strong boundary conditions
        u_ex_t_1, w_ex_t_1,p_ex_t_1 = problem.get_exact_sol_at_t(t_1_c)
        #bcu = [DirichletBC(V, u_ex_t_1, boundary_u_in)]
        #bcp = [DirichletBC(Q, p_ex_t_1, boundary_p_in)]
        bcu = [DirichletBC(V, u_ex_t_1, "on_boundary")]
        bcp = []

        # Define Storage Arrays
        outputs_arr = np.zeros((1 + n_t, 5))

        # Initial Functionals
        outputs_arr[0] = self.calculate_outputs()  # ||u_ex_t - u_t||,||p_ex_t - p_t||, H_t, H_ex_t, ||div(u_t)||

        print("Initial outputs for system: ", outputs_arr[0])

        # Variational problem definition

        self.weak_form(dt,problem)
        # Assemble LHS matrices
        A1 = assemble(self.a1)
        A2 = assemble(self.a2)
        A3 = assemble(self.a3)

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ", self.pol_deg)
        print("==============")

        self.start_timing()

        # Time loop from t_1 onwards
        for t in tqdm(t_range[1:]):
            # Compute tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, self.u_t_1.vector(), b1, "gmres", "ilu")

            # Pressure correction
            b2 = assemble(self.L2)
            if len(bcp) == 0: normalize(b2)
            [bc.apply(A2, b2) for bc in bcp]
            solve(A2, self.p_t_1.vector(), b2, 'gmres', 'hypre_amg')
            if len(bcp) == 0: normalize(self.p_t_1.vector())

            # Velocity correction
            b3 = assemble(self.L3)
            [bc.apply(A3, b3) for bc in bcu]
            solve(A3, self.u_t_1.vector(), b3, "gmres", "ilu")

            self.u_t.assign(self.u_t_1)
            self.p_t.assign(self.p_t_1)

            t_c.assign(float(t_c) + dt)
            t_1_c.assign(float(t_1_c) + dt)

            outputs_arr[self._ts] = self.calculate_outputs()
            self.update(problem, t)

        plt.subplot(2,2, 1)
        plt.bar(t_range, outputs_arr[:, 0], width=float(dt) / 2)
        plt.title("L2 error of u_t")
        plt.subplot(2, 2, 2)
        plt.bar(t_range, outputs_arr[:, 1], width=float(dt) / 2)
        plt.title("L2 error of p_t")
        plt.subplot(2, 2, 3)
        plt.plot(t_range, outputs_arr[:, 2:4])
        plt.legend(['H_t', 'H_ex_t'])
        plt.subplot(2, 2, 4)
        plt.plot(t_range,outputs_arr[:, 4])
        plt.title("divergence error of vector field")
        plt.show()

# Define strain-rate tensor
def epsilon(u):
    "Return symmetric gradient."
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p,mu):
    "Return stress tensor."
    return 2 * mu * epsilon(u) - p * Identity(len(u))


def boundary_u_in(x, on_boundary):
    return on_boundary and not near(x[0],1.0)

def boundary_p_in(x, on_boundary):
    return on_boundary and near(x[0],1.0)