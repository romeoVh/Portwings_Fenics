from dolfin import *
import matplotlib.pyplot as plt

# Create mesh
mesh = UnitSquareMesh(32, 32)
n_vec = FacetNormal(mesh)

# Define source functions
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=3)
g = Expression("sin(5.0*x[0])",degree=3)
u_0 = Expression("(0.0)",degree=3)

# Polynomial degree
pol_deg = 2

########################################################################
#                        Primal formulation
# ######################################################################
# Define function spaces and mixed (product) space
P_1 = FiniteElement("N1curl",mesh.ufl_cell(), pol_deg)
P_0 = FiniteElement("CG",mesh.ufl_cell(), pol_deg)
P = MixedElement([P_1,P_0])

W = FunctionSpace(mesh,P)

# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

# Define variational form
a = (dot(sigma, tau) + dot(grad(u), tau) + dot(sigma, grad(v)))*dx
L = - f*v*dx - g*v*ds

# Define essential BC on Gamma_D
def boundary_D(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
bc = DirichletBC(W.sub(1), u_0, boundary_D)

# Compute solution
w = Function(W)
solve(a == L, w, bc)
(sigma, u) = w.split()

# Plot sigma and u
plt.subplot(2,2,1)
s_plot = plot(sigma)
plt.title("sigma")
plt.colorbar(s_plot)
plt.subplot(2,2,2)
u_plot = plot(u)
plt.title("u")
plt.colorbar(u_plot)


########################################################################
#                        Dual formulation
# ######################################################################
# Define function spaces and mixed (product) space
P_n1 = FiniteElement("RT",mesh.ufl_cell(), pol_deg) # n=2 in this example
P_n = FiniteElement("DG",mesh.ufl_cell(), pol_deg-1)
P_d = MixedElement([P_n1,P_n])

W_d = FunctionSpace(mesh,P_d)

# Define trial and test functions
(sigma, u) = TrialFunctions(W_d)
(tau, v) = TestFunctions(W_d)

# Define variational form
a = (dot(sigma, tau) - div(tau)*u + div(sigma)*v)*dx
L = f*v*dx - u_0*dot(tau,n_vec)*ds

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        super().__init__(**kwargs) # This part is new!
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = -sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

G = BoundarySource(mesh, degree=3)

# Define essential BC on Gamma_N
def boundary_N_1(x):
    return x[1] < DOLFIN_EPS

def boundary_N_2(x):
    return x[1] > 1.0 - DOLFIN_EPS

bc = [DirichletBC(W_d.sub(0), G, boundary_N_1),DirichletBC(W_d.sub(0), G, boundary_N_2)]

# Test Implementation of essential boundary condition:
g_vec_exp_bot = ("0.0","sin(5.0*x[0])")
g_vec_bot = Expression(g_vec_exp_bot,degree=3)
g_vec_exp_top = ("1.0e6","-sin(5.0*x[0])")
g_vec_top = Expression(g_vec_exp_top,degree=3)
#bc = [DirichletBC(W_d.sub(0), g_vec_bot, boundary_N_1),DirichletBC(W_d.sub(0), g_vec_top, boundary_N_2)]


# Compute solution
w = Function(W_d)
solve(a == L, w, bc)
(sigma_d, u_d) = w.split()

# Plot sigma and u
plt.subplot(2,2,3)
s_plot = plot(sigma_d)
plt.title("sigma_d")
plt.colorbar(s_plot)
plt.subplot(2,2,4)
u_plot = plot(u_d)
plt.title("u_d")
plt.colorbar(u_plot)


plt.show()