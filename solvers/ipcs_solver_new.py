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

    def calculate_outputs(self,problem):
        # Calculates problem specific outputs in addition to H_t & ||div(u_t)||
        prob_out = problem.calculate_outputs(self.u_t,None,self.p_t)
        H_t = assemble(self.H_t)
        divU = assemble(self.div_u_t)
        return np.append(prob_out,np.array([H_t,divU]))

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

        # # Set strong boundary conditions
        bcu, bcw, bcp = problem.boundary_conditions(V, None, Q, t_1_c)
        # Question: Should all boundary conditions be at same time t+1 ?

        # Initialize solver outputs
        self.H_t = 0.5 * (inner(self.u_t, self.u_t) * dx) # Energy of System
        self.div_u_t = div(self.u_t)* dx # Divergence of velocity

        # Initialize problem outputs
        num_outputs_problem =  problem.init_outputs(t_c)

        # Define Storage Arrays
        num_outputs = 2
        self.outputs_arr = np.zeros((1 + n_t, num_outputs_problem+num_outputs))

        # Initial Outputs at t=0
        self.outputs_arr[0] = self.calculate_outputs(problem)
        print("Initial outputs for system: ", self.outputs_arr[0])
        # If problem has exact solution then outputs are:
        # ||u_ex_t - u_t||, N/A, ||p_ex_t - p_t||, H_ex_t, H_t, ||div(u_t)||
        # else the outputs are
        # H_t, ||div(u_t)||

        # Variational problem definition
        self.weak_form(dt,problem)
        # Assemble LHS matrices
        A1 = assemble(self.a1)
        A2 = assemble(self.a2)
        A3 = assemble(self.a3)

        print("Computation of the solution with # of DOFs: " + str(num_dof) + ", and deg: ", self.pol_deg)
        print("============================")

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

            # Update previous values
            self.u_t.assign(self.u_t_1)
            self.p_t.assign(self.p_t_1)

            # Update time variables
            t_c.assign(float(t_c) + dt)
            t_1_c.assign(float(t_1_c) + dt)

            # Calculate solver and problem outputs
            self.outputs_arr[self._ts] = self.calculate_outputs(problem)
            self.update(problem, t)

# Define strain-rate tensor expresion
def epsilon(u):
    "Return symmetric gradient."
    return sym(nabla_grad(u))

# Define stress tensor expresion
def sigma(u, p,mu):
    "Return stress tensor."
    return 2 * mu * epsilon(u) - p * Identity(len(u))

