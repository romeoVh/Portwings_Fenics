__author__ = "Anders Logg <logg@simula.no>"
__date__ = "2008-04-03"
__copyright__ = "Copyright (C) 2008-2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from dolfin import *
from time import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class SolverBase:
    "Base class for all solvers."

    def __init__(self, options):
        # Store options
        self.options = options

        # Reset some solver variables
        self._time = None
        self._cputime = 0.0
        self._ts = 1 # time step counter

        self.pol_deg = options["pol_deg"]


        # Reset files for storing solution


        # Reset storage for functional values and errors


    def prefix(self, problem):
        "Return file prefix for output files"
        p = problem.__module__.split(".")[-1].lower()
        s = self.__module__.split(".")[-1].lower()
        date_time_obj = datetime.now()
        time_stamp = date_time_obj.strftime("%Y_%m_%d_%H_%M_%S")
        return 'results/' + s + '/' + p + '/' + time_stamp

    def start_timing(self):
        """Start timing, will be paused automatically during update
        and stopped when the end-time is reached."""
        self._time = time()

    def solve(self, problem):
        "Solve problem"
        raise NotImplementedError

    def timestep(self, problem):
        "Return time step and time range for problem."
        return problem.dt, problem.n_t, problem.t_vec

    def update(self, problem, t):
        "Update problem at time t"

        # Add to accumulated CPU time
        timestep_cputime = time() - self._time
        self._cputime += timestep_cputime

        # Update problem
        problem.update_problem(t)

        # Evaluate functional and error
        # Store values
        # -->> Hasn't been used so far but was present in online code

        # Increase time step and record current time
        self._ts += 1
        self._time = time()

    def end_of_loop(self, problem, t):
        pass