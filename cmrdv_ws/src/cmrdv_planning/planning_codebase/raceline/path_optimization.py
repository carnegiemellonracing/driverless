'''
This file introduces the PathOptimizer class, used to determine an optimize path to track.
We are using a very general optimizer, that is not taking into account the structure (especially the sparsity)
of our optimization problem. Tools like GEKKO leveraging additional knowledge about the nature of the optimization
may result in better results.

We are using pytorch autodifferentiation, which is used by the Minimizer class to feed the scipy solver using
precise values for the hessians and jacobians.
'''

from dataclasses import dataclass
import math as math, numpy as np, torch, warnings
import cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.frenet as frenet
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline import Spline
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.optimizer.minimize import Minimizer
from typing import List, Callable

# from typing import TypeAlias    Python 3.10 only

# Path: TypeAlias = list[Spline]    Python 3.10 only


def diff_poly(coeffs: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    '''
    Create a differential polynomial function from its coefficients
    '''
    degree = len(coeffs[0])
    coeffs = coeffs[:, None, ...].detach() # trick to parallelize dot products
    def torch_poly(x: torch.Tensor) -> torch.Tensor:          
        powers = torch.vander(x, N=degree).double().detach() # is detach really working as a fix to BackwardPass not working?
        powers = powers[..., None] # trick to parallelize dot products

        results = torch.matmul(
            coeffs,
            powers
        )

        return torch.squeeze(results)

    return torch_poly


@dataclass
class Parameters:
    m: float # mass of the car

    lf: float
    lr: float
    
    track_width: float

# Useful decorators

constrs_registry = []
def constraint(func):
    def modified(*args, **kwargs):
        d = func(*args, **kwargs)
        fun = d['fun']
        return {'obj': d, 'fun': fun, 'name': func.__name__}

    constrs_registry.append(modified)

    return modified

class PathOptimizer(Minimizer):
    def __init__(self, reference_path: List[Spline], cumulative_lengths, delta_progress=10):
        super().__init__()

        assert(sorted(reference_path) == reference_path) # Check if it is sorted properly
        print("ref here: ")
        print(reference_path)
        #assert(all([np.all(reference_path[i].points[-1] == reference_path[i+1].points[0]) for i in range(len(reference_path) - 1)])) # Check if it is a continuous path

        self.reference_path = reference_path

        self.diff1 = torch.tensor(np.array([spline.first_der.coeffs for spline in reference_path]))
        self.diff2 = torch.tensor(np.array([spline.second_der.coeffs for spline in reference_path]))

        self.cumulative_lengths = torch.from_numpy(cumulative_lengths).contiguous()

        # (N+1)\Delta s = length
        self.n_steps = math.ceil(cumulative_lengths[-1]/delta_progress)
        self.delta_progress = delta_progress

        self.parameters = Parameters(10, 5, 5, 10)

        torch.autograd.set_detect_anomaly(True)

        # Compute at the end after initialization
        self.constraints = [f(self) for f in constrs_registry]
        #del constrs_registry

    def get_curvature(self, progress):
        # Get curvature
        # We clamp to avoid case when searchsorted returns last index (if greather than total length)
        # right=True not working for some reason
        index = torch.clamp_max(torch.searchsorted(self.cumulative_lengths, progress), len(self.cumulative_lengths) - 1)

        curvatures = frenet.get_curvature(
            diff_poly(self.diff1[index]),
            diff_poly(self.diff2[index]),
            torch.where(index > 0, (progress - self.cumulative_lengths[index-1]).to(torch.float32), progress.to(torch.float32)) # error float instead of double
        )

        return curvatures

    def interpolate_raceline(self, progress, previous_index, precision=20):
        '''
        Find a point on the raceline approximately corresponding to a given progress

        Inputs:
        -------------
        progress: float
            Targeted progress on the raceline

        precision: int
            Number of points used to get approximation for a specific spline

        Output:
        -------------
        (point, splines, progress): Tuple[np.ndarray[float], Spline, float]
            point: point on the raceline
            splines: Spline object describing the spline of the found point
            progress: exact value of the progress
        '''

        index = previous_index
        if index is None:
            index = np.searchsorted(self.cumulative_lengths, progress)
        else:
            n = len(self.cumulative_lengths)
            while self.cumulative_lengths[index] < progress and index < n:
                index += 1

            if index >= n:
                exit("Unreachable progress: the progress wanted is greater than the total length of the reference path")

        spline = self.reference_path[index]

        delta = progress if index == 0 else progress - self.cumulative_lengths[index-1]

        # local point is the point represented in the spline frame
        point, length, local_point = spline.along(delta, precision=precision)

        return (
            point,
            spline,
            local_point,
            progress - delta + length,
            index
        )

    def initial_states(self, progress_steps):
        n = torch.zeros(progress_steps.shape)
        mu = torch.zeros(progress_steps.shape)
        vx = torch.zeros(progress_steps.shape) + 0.1 # initial speed to avoid division by zero
        vy = torch.zeros(progress_steps.shape) + 0.1 # initial speed to avoid division by zero
        r = torch.zeros(progress_steps.shape)
        delta = torch.zeros(progress_steps.shape)
        T = torch.zeros(progress_steps.shape)

        states = torch.stack((progress_steps, n, mu, vx, vy, r, delta, T), dim=1)

        return states
    
    def initial_controls(self, progress_steps):
        return torch.zeros((self.n_steps, 2), requires_grad=True)

    def extract(self, variables):
        return variables[:, :8], variables[:, 8:]

    def continuous_dynamics(self, states, controls):
        # States: (1320, 8)
        # (progress_steps, n, mu, vx, vy, r, delta, T)

        s = states[:, 0]
        n = states[:, 1]
        mu = states[:, 2]
        vx = states[:, 3]
        vy = states[:, 4]
        r = states[:, 5]
        delta = states[:, 6]
        T = states[:, 7]

        # TODO: Change these
        Fx = 1
        FyF = 2
        FyR = 1
        Iz = 1

        ptv = 2 # change this, gain of the system
        rt = torch.tan(delta) * vx / (self.parameters.lf + self.parameters.lr)
        Mtv = ptv * (rt - r)

        curvature = self.get_curvature(s)

        s_dot = (vx * torch.cos(mu) - vy * torch.sin(mu))/(1 - n * curvature)
        n_dot = vx * torch.sin(mu) - vy * torch.cos(mu)
        mu_dot = r - curvature * s_dot.clone()
        vx_dot = (Fx - FyF * torch.sin(delta) + self.parameters.m * vy * r)/self.parameters.m
        vy_dot = (FyR + FyF * torch.cos(delta) - self.parameters.m * vx * r)/self.parameters.m
        r_dot = (FyF * self.parameters.lf * torch.cos(delta) - FyR * self.parameters.lr + Mtv)/Iz
        delta_dot = controls[:, 0]
        T_dot = controls[:, 1]

        return torch.stack((
            s_dot,
            n_dot,
            mu_dot,
            vx_dot,
            vy_dot,
            r_dot,
            delta_dot,
            T_dot
        ), dim=1)

    def discrete_dynamics(self, states, controls):
        # Forward Euler integration of the dynamics

        return states + self.delta_progress * self.continuous_dynamics(states, controls)

    def cost(self, variables):
        # States: (1320, 8)
        # (progress_steps, n, mu, vx, vy, r, delta, T)

        states, controls = self.extract(variables)

        s = states[:, 0]
        n = states[:, 1]
        mu = states[:, 2]
        vx = states[:, 3]
        vy = states[:, 4]

        # Time as a cost
        y = self.delta_progress * 1/self.continuous_dynamics(states, controls)[:, 0]

        #y += # first regularizer
        #y += # second regularizer
        return y.sum()
    
    #@constraint
    def dynamics_constraint(self):
        def constraint(variables):
            # States: (1320, 8)
            # (progress_steps, n, mu, vx, vy, r, delta, T)

            states, controls = self.extract(variables)

            computed = states[1::]
            rollout = self.discrete_dynamics(states[:-1:], controls[:-1:])

            return (computed - rollout).view((-1,)) # multiply this by 10 to add weight to the constraint?
        
        return dict(
            fun=(lambda v: constraint(v).square().sum()),
            lb=0, ub=0
        )

    #@constraint
    def closed_loop_constraint(self):
        def constraint(variables):
            # States: (1320, 8)
            # (progress_steps, n, mu, vx, vy, r, delta, T)

            states, controls = self.extract(variables)

            end = self.discrete_dynamics(states[-1][None, ...], controls[-1][None, ...])

            # we want the progress to be 0 at the end
            return (end[0][0] - self.cumulative_lengths[-1]).view((-1,))

        return dict(
            fun=(lambda v: constraint(v).square().sum()),
            lb=0, ub=0
        )

    ## Controls constraint

    @constraint
    def delta_constraint(self):
        delta_min = -10
        delta_max = 10

        def constraint(variables):
            # (progress_steps, n, mu, vx, vy, r, delta, T)

            states, controls = self.extract(variables)

            delta = controls[:, 0]
            
            return delta - torch.clamp(delta, min=delta_min, max=delta_max)

        return dict(
            fun=(lambda v: constraint(v).square().sum()),
            lb=0, ub=0,
            jacobian=True, hessian=True,# How to not compute constraint curvature?
            #finite_diff_jac_sparsity=[[i, 6] for i in range(self.n_steps)] # shape=(343, 2)
        )
    
    @constraint
    def T_constraint(self):
        T_min = -10
        T_max = 10
        
        def constraint(variables):
            # (progress_steps, n, mu, vx, vy, r, delta, T)

            states, _ = self.extract(variables)
            
            T = states[:, 7]

            return T - torch.clamp(T, min=T_min, max=T_max)

        # Not ideal, a better thing would be to have lower and upper bounds
        return dict(
            fun=(lambda v: constraint(v).square().sum()),
            lb=0, ub=0
        )

    def track_constraint(self, variables):
        states, controls = self.extract(variables)
        n_vals = states[:, :1] #Column of n values
        n_vals = torch.tensor.sub(n_vals, self.parameters.track_width/2)
        pos_indecies = torch.nonzero(n_vals)
        acc = 0
        for i in pos_indecies:
            acc += n_vals[i]
        return acc

    def friction_ellipse_constraint(self):
        pass

    def optimize(self):
        total_length = self.cumulative_lengths[-1]
        progress_steps = torch.linspace(0, total_length, self.n_steps)

        init = torch.cat((self.initial_states(progress_steps), self.initial_controls(progress_steps)), dim=1)

        # Minimize cost under constraints
        result = self.minimize(init, tol=1e-2, disp=1, max_iter=40)

        if not result.success:
            warnings.warn("Path optimization failed, outputs may not be optimal")

        # Solution contains states and controls
        solution = result.x
        print("got here")
        return solution.numpy(), solution