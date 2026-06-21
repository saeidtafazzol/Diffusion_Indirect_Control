import os
import numpy as np
import casadi as ca


class DynamicsBase:
    def __init__(self, n_x=7):
        self.backend = 'casadi'
        self.n_x = n_x

        self.temp_dir = "temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.states = ca.MX.sym('states', self.n_x)
        self.costates = ca.MX.sym('costates', self.n_x)
        self.delta = ca.MX.sym('delta')
        self.alpha = ca.MX.sym('alpha', 3)

        # Common parameters
        self.c = ca.MX.sym('c')
        self.t_max = ca.MX.sym('t_max')
        self.eps = ca.MX.sym('eps')

        # Symbolic expressions
        self.state_dot_expr = None
        self.costate_dot_expr = None
        self.hamiltonian_expr = None

        # Optimal-control-substituted expressions
        self.state_dot_opt = None
        self.costate_dot_opt = None
        self.augmented_dot_opt = None
        self.hamiltonian_opt = None

        # Parameter-substituted expressions
        self.state_dot_sub = None
        self.costate_dot_sub = None
        self.augmented_dot_sub = None
        self.hamiltonian_sub = None

        self.params = {}

        self.populate_state_dot()

    def populate_state_dot(self):
        raise NotImplementedError("Subclasses must implement populate_state_dot()")

    def configure_params(self, **param_values):
        params = {}
        if 'c' in param_values:
            params['c'] = param_values['c']
        if 't_max' in param_values:
            params['t_max'] = param_values['t_max']
        if 'eps' in param_values:
            params['eps'] = param_values['eps']
        return params

    def _substitute_params(self, expr):
        if hasattr(self, 'params') and 'c' in self.params:
            expr = ca.substitute(expr, self.c, self.params['c'])
        if hasattr(self, 'params') and 't_max' in self.params:
            expr = ca.substitute(expr, self.t_max, self.params['t_max'])
        if hasattr(self, 'params') and 'eps' in self.params:
            expr = ca.substitute(expr, self.eps, self.params['eps'])
        return expr

    def set_params(self, **params):
        self.params = self.configure_params(**params)
        self.state_dot_sub = self._substitute_params(self.state_dot_opt)
        self.costate_dot_sub = self._substitute_params(self.costate_dot_opt)
        self.augmented_dot_sub = self._substitute_params(self.augmented_dot_opt)
        self.hamiltonian_sub = self._substitute_params(self.hamiltonian_opt)

    def _compile_function(self, func_name, expr, filename_base):
        inputs = [self.states, self.costates]
        func_aux = ca.Function(func_name, inputs, [expr])
        current_dir = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            c_filename = f"{filename_base}.c"
            so_filename = f"{filename_base}.so"
            func_aux.generate(c_filename)
            os.system(f"gcc -fPIC -shared {c_filename} -o {so_filename}")
            os.chdir(current_dir)
            return ca.external(func_name, os.path.join(self.temp_dir, so_filename))
        except Exception as e:
            os.chdir(current_dir)
            raise e


class TwoBodyCartesian(DynamicsBase):
    """Two-body min-fuel dynamics in Cartesian coordinates (normalized units).

    State: [r_x, r_y, r_z, v_x, v_y, v_z, m]  (m starts at 1)
    Costate: [lambda_r (3), lambda_v (3), lambda_m]  (7 components)
    """

    def __init__(self, n_x=7):
        self.mu = ca.MX.sym('mu')
        super().__init__(n_x)

    def populate_state_dot(self):
        r0, r1, r2 = self.states[0], self.states[1], self.states[2]
        v0, v1, v2 = self.states[3], self.states[4], self.states[5]
        m = self.states[6]  # mass in normalized units (starts at 1)

        r_mag = ca.sqrt(r0**2 + r1**2 + r2**2)
        g_vec = -self.mu / r_mag**3 * ca.vertcat(r0, r1, r2)

        lambda_v_mag = ca.sqrt(
            self.costates[3]**2 + self.costates[4]**2 + self.costates[5]**2
        )

        # State dynamics
        r_dot = ca.vertcat(v0, v1, v2)
        v_dot = g_vec + self.delta * (self.t_max / m) * self.alpha
        m_dot = -self.delta * self.t_max / self.c
        self.state_dot_expr = ca.vertcat(r_dot, v_dot, m_dot)

        # Hamiltonian: H = L + lambda^T f
        cost = self.t_max / self.c * self.delta
        self.hamiltonian_expr = cost + ca.dot(self.costates, self.state_dot_expr)

        # Costate dynamics: lambda_dot = -dH/dx
        self.costate_dot_expr = -ca.gradient(self.hamiltonian_expr, self.states)

        # Optimal controls via Pontryagin
        S = self.c * lambda_v_mag / m + self.costates[6] - 1
        p_vec = -self.costates[3:6]
        p_norm = ca.norm_2(p_vec)

        optimal_alpha = p_vec / p_norm
        optimal_delta = (1 + S / ca.sqrt(S**2 + self.eps)) * 0.5

        # Substitute optimal controls
        self.state_dot_opt = ca.substitute(
            ca.substitute(self.state_dot_expr, self.alpha, optimal_alpha),
            self.delta, optimal_delta,
        )
        self.costate_dot_opt = ca.substitute(
            ca.substitute(self.costate_dot_expr, self.alpha, optimal_alpha),
            self.delta, optimal_delta,
        )
        self.hamiltonian_opt = ca.substitute(
            ca.substitute(self.hamiltonian_expr, self.alpha, optimal_alpha),
            self.delta, optimal_delta,
        )
        self.augmented_dot_opt = ca.vertcat(self.state_dot_opt, self.costate_dot_opt)

    def configure_params(self, **param_values):
        params = super().configure_params(**param_values)
        if 'mu' in param_values:
            params[self.mu] = param_values['mu']
        return params

    def _substitute_params(self, expr):
        sub_expr = ca.substitute(expr, self.mu, self.params[self.mu])
        sub_expr = super()._substitute_params(sub_expr)
        return sub_expr
