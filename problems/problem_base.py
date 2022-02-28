from fenics import *

from numpy import linspace, array
import matplotlib.pyplot as plt
from matplotlib import interactive
from mshr import *

class ProblemBase:
    "Base class for all problems."

    def __init__(self, options):

        # Store options
        self.options = options

        self.n_el = options["n_el"]  # Number of spatial elements
        self.t_fin = options["t_fin"]  # Final time
        self.n_t = options["n_t"]  # Number of temporal elements

        # Parameters must be defined by subclass
        self.mesh = None
        self.dimM = None
        self.n_ver = None
        self.dt = None
        self.t_vec = None
        self.output_location = ''

    def init_mesh(self,show_mesh = False):
        self.mesh.init()
        self.dimM = self.mesh.topology().dim()
        print("Initialized Mesh with cells,faces,edges,vertices: ",
              [self.mesh.num_cells(), self.mesh.num_faces(), self.mesh.num_edges(), self.mesh.num_vertices()])
        self.n_ver = FacetNormal(self.mesh)
        if show_mesh:
            plt.ion()
            plot(self.mesh,alpha=0.9)
            plt.pause(0.01)
            #plt.show()

        #plt.show(block=False)

    def structured_time_grid(self):
        self.dt = self.t_fin / self.n_t
        self.t_vec = linspace(0, self.t_fin, self.n_t + 1)

    def update_problem(self, t):
        "Update problem-specific at time t"
        pass


    def initial_conditions(self, V, Q):
        pass

    def init_outputs(self, t_c):
        # no outputs
        return []

    def calculate_outputs(self,exact_arr,u_t,w_t,p_t):
        return array([])

    def convert_sym(self, name, fun, show_func=False):
        import sympy as sym
        fun_code = sym.printing.ccode(fun)
        if (show_func):
            print('Code of ', name, ' is: ', fun_code)
        return fun_code
