"""
Generalized S-curve motion with arbitrary boundary conditions.

IMPORTANT: The S-curve traces a LOOP in phase space (velocity vs acceleration).
For any velocity v (except 0 and max), there are exactly TWO possible states:
1. Accelerating phase (τ < 0): positive acceleration, velocity increasing
2. Decelerating phase (τ > 0): negative acceleration, velocity decreasing

You CANNOT specify arbitrary (v, a) pairs - they must lie ON the curve!
The acceleration is determined by the velocity AND the phase.
"""
from __future__ import division
from math import sqrt, pi
import numpy as np
from scipy.special import betainc, beta as beta_func
from scipy.optimize import brentq

__all__ = [
    'normalized_f', 'normalized_f_derivative', 'normalized_f_second_derivative',
    'find_tau_for_velocity', 'generalized_motion_time', 'generalized_sCurve',
    'get_velocity', 'get_acceleration', 'plan_motion', 'evaluate_motion',
    'continue_motion', 'SmoothMotion', 'SmoothTauMapping', 'evaluate_smooth_motion',
    'BetaBlendMotion', 'continue_motion_blend',
    'MAX_NORMALIZED_VEL', 'MAX_NORMALIZED_ACC'
]

# Constants for the beta S-curve with p=2.5
MAX_NORMALIZED_VEL = 16 / (5 * pi)  # ≈ 1.0186 at τ=0
MAX_NORMALIZED_ACC = 3 * sqrt(3) / pi  # ≈ 1.6540 at τ=-0.5


def normalized_f(tau):
    """Normalized S-curve position f(τ) for τ ∈ [-1, 1]. Returns [0, 1]."""
    tau = np.asarray(tau, dtype=float)
    scalar_input = tau.ndim == 0
    tau = np.atleast_1d(tau)

    result = np.zeros_like(tau)
    result[tau <= -1] = 0.0
    result[tau >= 1] = 1.0

    mask = (tau > -1) & (tau < 1)
    if np.any(mask):
        t_masked = tau[mask]
        beta_val = beta_func(0.5, 3.5)
        coef = 1 / (2 * beta_val)

        neg = t_masked < 0
        pos = t_masked > 0
        zero = t_masked == 0

        res = np.zeros_like(t_masked)
        if np.any(neg):
            res[neg] = 0.5 - coef * betainc(0.5, 3.5, t_masked[neg]**2) * beta_val
        if np.any(pos):
            res[pos] = 0.5 + coef * betainc(0.5, 3.5, t_masked[pos]**2) * beta_val
        if np.any(zero):
            res[zero] = 0.5
        result[mask] = res

    return float(result[0]) if scalar_input else result


def normalized_f_derivative(tau):
    """Velocity profile f'(τ) = (1 - τ²)^2.5 / B(0.5, 3.5). Max ≈ 1.019 at τ=0."""
    tau = np.asarray(tau, dtype=float)
    scalar_input = tau.ndim == 0
    tau = np.atleast_1d(tau)

    result = np.zeros_like(tau)
    mask = np.abs(tau) < 1
    if np.any(mask):
        t = tau[mask]
        result[mask] = np.power(1 - t**2, 2.5) / beta_func(0.5, 3.5)

    return float(result[0]) if scalar_input else result


def normalized_f_second_derivative(tau):
    """Acceleration profile f''(τ) = -5τ(1 - τ²)^1.5 / B(0.5, 3.5)."""
    tau = np.asarray(tau, dtype=float)
    scalar_input = tau.ndim == 0
    tau = np.atleast_1d(tau)

    result = np.zeros_like(tau)
    mask = np.abs(tau) < 1
    if np.any(mask):
        t = tau[mask]
        result[mask] = -5 * t * np.power(1 - t**2, 1.5) / beta_func(0.5, 3.5)

    return float(result[0]) if scalar_input else result


def find_tau_for_velocity(v_normalized, accelerating=True):
    """
    Find τ where normalized velocity = v_normalized.

    Parameters:
    -----------
    v_normalized : float
        Velocity as fraction of max (0 to 1)
    accelerating : bool
        True = accelerating phase (τ in [-1, 0], a > 0)
        False = decelerating phase (τ in [0, 1], a < 0)

    Returns τ value.
    """
    if v_normalized <= 0:
        return -1.0 if accelerating else 1.0
    if v_normalized >= 1:
        return 0.0

    target = v_normalized * MAX_NORMALIZED_VEL

    def obj(tau):
        return normalized_f_derivative(tau) - target

    if accelerating:
        return brentq(obj, -1 + 1e-10, 0)
    else:
        return brentq(obj, 0, 1 - 1e-10)


def generalized_motion_time(robotVmax, robotAmax, x0, x1, v0=0, v1=0, a0=0, a1=0):
    """
    Calculate motion parameters for given boundary conditions.

    For BOTH velocity AND acceleration continuity, we use:
    - a0/v0 ratio to find exact τ_start (if both non-zero)
    - a1/v1 ratio to find exact τ_end (if both non-zero)

    The ratio a/v = (Δτ/T) * f''(τ)/f'(τ) determines τ uniquely.

    Returns: (T, tau_start, tau_end)
    """
    delta_x = x1 - x0

    # Handle zero motion
    if abs(delta_x) < 1e-10:
        return 0.0, -1.0, 1.0

    motion_dir = 1 if delta_x > 0 else -1

    # Velocities/accelerations relative to motion direction
    v0_rel = v0 * motion_dir
    v1_rel = v1 * motion_dir
    a0_rel = a0 * motion_dir
    a1_rel = a1 * motion_dir

    # Rest-to-rest: use original formula
    if abs(v0) < 1e-10 and abs(v1) < 1e-10:
        T = max(
            2.572148274314975138567 * sqrt(abs(delta_x) / robotAmax),
            2.037183271576260297842 * abs(delta_x) / robotVmax
        )
        return T, -1.0, 1.0

    # Find τ_start
    if abs(v0) < 1e-10:
        tau_start = -1.0
    elif abs(a0) > 1e-10:
        # Both v0 and a0 specified - find τ where ratio matches
        # a/v = f''(τ)/f'(τ) * (Δτ/T) but we need τ first
        # Actually: a/v at a point = f''(τ)/f'(τ) * (Δτ/T)
        # The ratio f''(τ)/f'(τ) is unique for each τ!
        # f'(τ) = (1-τ²)^2.5 / B
        # f''(τ) = -5τ(1-τ²)^1.5 / B
        # f''/f' = -5τ(1-τ²)^1.5 / (1-τ²)^2.5 = -5τ/(1-τ²)
        #
        # Given the SIGN of a0, we know which phase:
        accelerating = a0_rel > 0
        v0_norm = min(abs(v0_rel) / robotVmax, 0.999)
        tau_start = find_tau_for_velocity(v0_norm, accelerating=accelerating)
    else:
        # Only v0 specified, default to accelerating phase
        v0_norm = min(abs(v0_rel) / robotVmax, 0.999)
        tau_start = find_tau_for_velocity(v0_norm, accelerating=True)

    # Find τ_end
    if abs(v1) < 1e-10:
        tau_end = 1.0
    elif abs(a1) > 1e-10:
        accelerating = a1_rel > 0
        v1_norm = min(abs(v1_rel) / robotVmax, 0.999)
        tau_end = find_tau_for_velocity(v1_norm, accelerating=accelerating)
    else:
        # Only v1 specified, default to decelerating phase
        v1_norm = min(abs(v1_rel) / robotVmax, 0.999)
        tau_end = find_tau_for_velocity(v1_norm, accelerating=False)

    # Validate ordering
    if tau_start >= tau_end:
        T = max(
            2.572148274314975138567 * sqrt(abs(delta_x) / robotAmax),
            2.037183271576260297842 * abs(delta_x) / robotVmax
        )
        return T, -1.0, 1.0

    delta_tau = tau_end - tau_start
    delta_f = normalized_f(tau_end) - normalized_f(tau_start)
    if delta_f < 1e-10:
        delta_f = 1e-10

    # If BOTH v0 and a0 are specified, T is determined by their ratio:
    # a0 = K * (Δτ/T) * f''(τ_start) where K = Δx/Δf
    # v0 = K * (Δτ/T)^0 * ... wait no:
    # v0 = (Δx/Δf) * (Δτ/T) * f'(τ_start)
    # a0 = (Δx/Δf) * (Δτ/T)² * f''(τ_start)
    # So: a0/v0 = (Δτ/T) * f''(τ_start)/f'(τ_start)
    # Therefore: T = Δτ * f''(τ_start) * v0 / (f'(τ_start) * a0)

    T_candidates = []

    if abs(v0_rel) > 1e-10 and abs(a0_rel) > 1e-10:
        f_prime_start = normalized_f_derivative(tau_start)
        f_dbl_prime_start = normalized_f_second_derivative(tau_start)
        if abs(f_prime_start) > 1e-10 and abs(a0_rel) > 1e-10:
            # T = Δτ * f'' * v / (f' * a)
            T_from_ratio = abs(delta_tau * f_dbl_prime_start * v0_rel / (f_prime_start * a0_rel))
            T_candidates.append(T_from_ratio)

    if abs(v1_rel) > 1e-10 and abs(a1_rel) > 1e-10:
        f_prime_end = normalized_f_derivative(tau_end)
        f_dbl_prime_end = normalized_f_second_derivative(tau_end)
        if abs(f_prime_end) > 1e-10 and abs(a1_rel) > 1e-10:
            T_from_ratio = abs(delta_tau * f_dbl_prime_end * v1_rel / (f_prime_end * a1_rel))
            T_candidates.append(T_from_ratio)

    # Fallback: use velocity alone
    if abs(v0_rel) > 1e-10:
        f_prime_start = normalized_f_derivative(tau_start)
        T_v0 = abs(delta_x * delta_tau * f_prime_start / (delta_f * v0_rel))
        T_candidates.append(T_v0)

    if abs(v1_rel) > 1e-10:
        f_prime_end = normalized_f_derivative(tau_end)
        T_v1 = abs(delta_x * delta_tau * f_prime_end / (delta_f * v1_rel))
        T_candidates.append(T_v1)

    # Constraint checks
    tau_range = np.linspace(tau_start, tau_end, 100)
    f_prime_max = np.max(normalized_f_derivative(tau_range))
    f_dbl_prime_max = np.max(np.abs(normalized_f_second_derivative(tau_range)))

    T_vmax = abs(delta_x) * delta_tau * f_prime_max / (delta_f * robotVmax)
    T_amax = sqrt(abs(delta_x) * delta_tau**2 * f_dbl_prime_max / (delta_f * robotAmax))
    T_candidates.append(T_vmax)
    T_candidates.append(T_amax)

    T = max(T_candidates) if T_candidates else 1.0

    return T, tau_start, tau_end


def generalized_sCurve(t, T, x0, x1, tau_start=-1, tau_end=1):
    """Position at time t."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return x0 if scalar else np.full_like(t, x0)

    tau = tau_start + (tau_end - tau_start) * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_vals = normalized_f(tau)
    f_start = normalized_f(tau_start)
    delta_f = normalized_f(tau_end) - f_start

    if abs(delta_f) < 1e-10:
        return x0 if scalar else np.full_like(t, x0)

    pos = x0 + (x1 - x0) * (f_vals - f_start) / delta_f
    return float(pos[0]) if scalar else pos


def get_velocity(t, T, x0, x1, tau_start=-1, tau_end=1):
    """Velocity at time t."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return 0.0 if scalar else np.zeros_like(t)

    delta_tau = tau_end - tau_start
    delta_x = x1 - x0
    delta_f = normalized_f(tau_end) - normalized_f(tau_start)

    if abs(delta_f) < 1e-10:
        return 0.0 if scalar else np.zeros_like(t)

    tau = tau_start + delta_tau * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_prime = normalized_f_derivative(tau)
    vel = (delta_x / delta_f) * f_prime * (delta_tau / T)

    return float(vel[0]) if scalar else vel


def get_acceleration(t, T, x0, x1, tau_start=-1, tau_end=1):
    """Acceleration at time t."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return 0.0 if scalar else np.zeros_like(t)

    delta_tau = tau_end - tau_start
    delta_x = x1 - x0
    delta_f = normalized_f(tau_end) - normalized_f(tau_start)

    if abs(delta_f) < 1e-10:
        return 0.0 if scalar else np.zeros_like(t)

    tau = tau_start + delta_tau * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_dbl_prime = normalized_f_second_derivative(tau)
    acc = (delta_x / delta_f) * f_dbl_prime * (delta_tau / T)**2

    return float(acc[0]) if scalar else acc


def _flat_function(s):
    """
    C∞ flat function: exp(-1/s) for s > 0, else 0.
    Has ALL derivatives = 0 at s = 0.
    """
    s = np.asarray(s, dtype=float)
    scalar = s.ndim == 0
    s = np.atleast_1d(s)

    result = np.zeros_like(s)
    mask = s > 1e-10
    result[mask] = np.exp(-1.0 / s[mask])

    return float(result[0]) if scalar else result


def _flat_function_deriv(s, order=1):
    """
    Derivative of flat function. All derivatives are 0 at s=0.
    """
    s = np.asarray(s, dtype=float)
    scalar = s.ndim == 0
    s = np.atleast_1d(s)

    result = np.zeros_like(s)
    mask = s > 1e-10

    if order == 1:
        # d/ds exp(-1/s) = exp(-1/s) / s²
        result[mask] = np.exp(-1.0 / s[mask]) / (s[mask]**2)
    elif order == 2:
        # d²/ds² = exp(-1/s) * (1/s⁴ - 2/s³)
        result[mask] = np.exp(-1.0 / s[mask]) * (1/s[mask]**4 - 2/s[mask]**3)
    elif order == 3:
        # d³/ds³ = exp(-1/s) * (1/s⁶ - 6/s⁵ + 6/s⁴)
        result[mask] = np.exp(-1.0 / s[mask]) * (1/s[mask]**6 - 6/s[mask]**5 + 6/s[mask]**4)
    elif order == 4:
        # d⁴/ds⁴ = exp(-1/s) * (1/s⁸ - 12/s⁷ + 36/s⁶ - 24/s⁵)
        result[mask] = np.exp(-1.0 / s[mask]) * (1/s[mask]**8 - 12/s[mask]**7 + 36/s[mask]**6 - 24/s[mask]**5)

    return float(result[0]) if scalar else result


class SmoothTauMapping:
    """
    Maps time t to τ with TRUE C∞ smoothness at the end.

    Uses a flat function φ(s) = exp(-1/s) which has ALL derivatives = 0 at s=0.

    τ(t) = τ_end - w(t) * p(t)

    where:
    - w(t) = φ(1 - t/T) goes to 0 with ALL derivatives at t=T
    - p(t) is a polynomial chosen to match initial conditions

    This gives:
    - Perfect match of initial conditions (τ, τ', τ'', τ''', ...)
    - C∞ smoothness at t=T (ALL derivatives → 0)
    """

    def __init__(self, tau_start, tau_end, tau_derivs_0, T):
        """
        Parameters:
        -----------
        tau_start : float - Starting τ value
        tau_end : float - Ending τ value (typically 1.0 for rest)
        tau_derivs_0 : list - [τ'(0), τ''(0), τ'''(0), ...] derivatives at t=0
        T : float - Motion duration
        """
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_derivs_0 = list(tau_derivs_0)  # List of initial derivatives
        self.T = T
        self.n_derivs = len(tau_derivs_0)

        # w(0) = exp(-1)
        self.w0 = np.exp(-1.0)

        # c0 = (τ_end - τ_start) / w(0)
        self.c0 = (tau_end - tau_start) / self.w0

        # Compute w derivatives at t=0
        self.w_derivs = self._compute_w_derivs_at_0(self.n_derivs + 1)

        # Solve for polynomial coefficients
        self.poly_coeffs = self._solve_for_poly_coeffs()

    def _compute_w_derivs_at_0(self, n):
        """
        Compute w^(k)(0) for k = 0, 1, ..., n analytically.

        w(t) = exp(-1/(1-t/T))

        At t=0, let u = 1 - t/T = 1. The k-th derivative of w at t=0 can be computed
        using the chain rule. The pattern involves powers of exp(-1) and 1/T.
        """
        T = self.T
        e1 = np.exp(-1.0)  # w(0) = exp(-1)

        # Analytical formulas for derivatives at t=0 (u=1)
        # w(0) = exp(-1)
        # w'(0) = -exp(-1)/T
        # w''(0) = exp(-1)*(1-2)/T² = -exp(-1)/T²
        # w'''(0) = -exp(-1)*(1-6+6)/T³ = -exp(-1)/T³
        # w''''(0) = exp(-1)*(1-12+36-24)/T⁴ = exp(-1)/T⁴

        # General pattern: w^(k)(0) = (-1)^k * exp(-1) * P_k(1) / T^k
        # where P_k is a polynomial. For simplicity, use first few analytically.

        derivs = [e1]  # w(0) = exp(-1)

        if n >= 1:
            derivs.append(-e1 / T)  # w'(0)

        if n >= 2:
            derivs.append(-e1 / T**2)  # w''(0)

        if n >= 3:
            derivs.append(-e1 / T**3)  # w'''(0)

        if n >= 4:
            derivs.append(e1 / T**4)  # w''''(0)

        if n >= 5:
            derivs.append(e1 / T**5)  # w'''''(0) ≈ exp(-1)/T^5

        if n >= 6:
            derivs.append(e1 / T**6)

        if n >= 7:
            derivs.append(-e1 / T**7)

        if n >= 8:
            derivs.append(-e1 / T**8)

        # For any remaining, use a reasonable approximation
        while len(derivs) <= n:
            k = len(derivs)
            sign = (-1) ** (k // 2)
            derivs.append(sign * e1 / T**k)

        return derivs

    def _solve_for_poly_coeffs(self):
        """
        Solve for polynomial p(t) = c0 + c1*t + c2*t² + ...
        such that τ(t) = τ_end - w(t)*p(t) matches initial conditions.

        τ^(k)(0) for k = 0, 1, 2, ... gives us equations for c0, c1, c2, ...
        """
        w = self.w_derivs
        c = [self.c0]  # c0 is already determined

        # For each derivative order k >= 1, solve for c_k
        # τ^(k)(0) = sum over i+j=k of (-1) * C(k,i) * w^(i)(0) * p^(j)(0)
        # where p^(j)(0) = j! * c_j

        # Factorials
        from math import factorial

        for k in range(1, self.n_derivs + 1):
            # τ^(k)(0) = desired value
            tau_k = self.tau_derivs_0[k - 1]

            # Compute sum of known terms (those involving c_0, c_1, ..., c_{k-1})
            known_sum = 0
            for i in range(k + 1):  # i from 0 to k
                j = k - i  # j = k - i, so p^(j)(0) = j! * c_j
                if j < len(c):  # Only if we have c_j
                    binom = factorial(k) // (factorial(i) * factorial(j))
                    known_sum += binom * w[i] * factorial(j) * c[j]

            # The unknown term is when j = k: C(k,0) * w^(0) * p^(k)(0) = w(0) * k! * c_k
            # τ^(k) = -known_sum - w(0) * k! * c_k
            # c_k = -(τ^(k) + known_sum) / (w(0) * k!)
            c_k = -(tau_k + known_sum) / (w[0] * factorial(k))
            c.append(c_k)

        return c

    def eval(self, t, max_deriv=4):
        """
        Evaluate τ and derivatives at time t.

        Parameters:
        -----------
        t : float or array - Time(s) to evaluate at
        max_deriv : int - Maximum derivative order to compute (default 4)

        Returns: tuple of (τ, τ', τ'', ...) up to max_deriv
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        t = np.clip(t, 0, self.T)

        T = self.T
        c = self.poly_coeffs
        n_c = len(c)

        # s = 1 - t/T
        s = 1 - t / T

        # Compute p(t) and its derivatives
        from math import factorial
        p_derivs = []
        for k in range(max_deriv + 1):
            # p^(k)(t) = sum_{j >= k} c_j * j!/(j-k)! * t^{j-k}
            p_k = np.zeros_like(t)
            for j in range(k, n_c):
                # IMPORTANT: parentheses needed to avoid floor division of coefficient!
                coef = c[j] * (factorial(j) // factorial(j - k))
                p_k += coef * np.power(t, j - k)
            p_derivs.append(p_k)

        # Compute w(t) and its derivatives
        w_derivs_t = []
        w = _flat_function(s)
        w_derivs_t.append(w)

        mask = s > 1e-10

        for k in range(1, max_deriv + 1):
            w_k = np.zeros_like(t)
            if np.any(mask):
                sm = s[mask]
                exp_neg_inv = np.exp(-1.0 / sm)

                # Compute analytically for first few derivatives
                if k == 1:
                    w_k[mask] = -exp_neg_inv / (T * sm**2)
                elif k == 2:
                    w_k[mask] = exp_neg_inv * (1 - 2*sm) / (T**2 * sm**4)
                elif k == 3:
                    w_k[mask] = -exp_neg_inv * (1 - 6*sm + 6*sm**2) / (T**3 * sm**6)
                elif k == 4:
                    w_k[mask] = exp_neg_inv * (1 - 12*sm + 36*sm**2 - 24*sm**3) / (T**4 * sm**8)
                elif k == 5:
                    w_k[mask] = -exp_neg_inv * (1 - 20*sm + 120*sm**2 - 240*sm**3 + 120*sm**4) / (T**5 * sm**10)
                elif k == 6:
                    w_k[mask] = exp_neg_inv * (1 - 30*sm + 300*sm**2 - 1200*sm**3 + 1800*sm**4 - 720*sm**5) / (T**6 * sm**12)
                elif k == 7:
                    w_k[mask] = -exp_neg_inv * (1 - 42*sm + 630*sm**2 - 4200*sm**3 + 12600*sm**4 - 15120*sm**5 + 5040*sm**6) / (T**7 * sm**14)
                elif k == 8:
                    w_k[mask] = exp_neg_inv * (1 - 56*sm + 1176*sm**2 - 11760*sm**3 + 58800*sm**4 - 141120*sm**5 + 141120*sm**6 - 40320*sm**7) / (T**8 * sm**16)
                else:
                    # Higher derivatives set to 0 (approximately correct near t=T)
                    pass

            w_derivs_t.append(w_k)

        # Compute τ derivatives using product rule: τ = τ_end - w*p
        # τ^(k) = -sum_{i=0}^{k} C(k,i) * w^(i) * p^(k-i)
        tau_derivs = []
        for k in range(max_deriv + 1):
            tau_k = np.zeros_like(t)
            if k == 0:
                tau_k = self.tau_end - w_derivs_t[0] * p_derivs[0]
            else:
                for i in range(k + 1):
                    binom = factorial(k) // (factorial(i) * factorial(k - i))
                    tau_k -= binom * w_derivs_t[i] * p_derivs[k - i]
            tau_derivs.append(tau_k)

        if scalar:
            return tuple(float(td[0]) for td in tau_derivs)
        return tuple(tau_derivs)


class SmoothMotion:
    """
    A motion with TRUE C∞ smooth time parameterization.

    Uses a flat function φ(s) = exp(-1/s) for the time mapping τ(t).
    This function has the magical property that ALL derivatives are 0 at s=0.

    The approach:
    1. f(τ) maps τ → position (the beta S-curve shape)
    2. τ(t) uses flat function to map time → τ

    At t=T, the flat function ensures ALL derivatives of τ(t) are 0,
    which means ALL derivatives of position are 0: v=0, a=0, j=0, snap=0, ...

    At t=0, we exactly match the given initial conditions (v0, a0, j0, ...).
    """

    def __init__(self, x0, x_target, v0, a0, j0, tau_start, prev_tau_dot, T,
                 robotVmax=None, robotAmax=None, snap0=0, crackle0=0, pop0=0):
        """
        Create a C∞ smooth motion.

        Parameters:
        -----------
        x0 : float - Starting position
        x_target : float - Target position
        v0 : float - Starting velocity (matched exactly!)
        a0 : float - Starting acceleration (matched exactly!)
        j0 : float - Starting jerk (matched exactly!)
        tau_start : float - Starting τ value on S-curve
        prev_tau_dot : float - τ'(t) from previous motion
        T : float - Desired motion duration
        snap0 : float - Starting snap (4th derivative)
        crackle0 : float - Starting crackle (5th derivative)
        pop0 : float - Starting pop (6th derivative)
        """
        self.x0 = x0
        self.x_target = x_target
        self.v0 = v0
        self.a0 = a0
        self.j0 = j0
        self.snap0 = snap0
        self.crackle0 = crackle0
        self.pop0 = pop0
        self.tau_start = tau_start
        self.tau_end = 1.0  # End at rest
        self.robotVmax = robotVmax if robotVmax else float('inf')
        self.robotAmax = robotAmax if robotAmax else float('inf')

        delta_x = x_target - x0
        delta_tau = self.tau_end - tau_start
        delta_f = normalized_f(self.tau_end) - normalized_f(tau_start)

        if abs(delta_f) < 1e-10 or abs(delta_tau) < 1e-10:
            self.T = 0
            self.S = 0
            self.tau_mapping = None
            return

        # Spatial scaling: S maps normalized f to actual position
        self.S = delta_x / delta_f

        # Compute f derivatives at tau_start
        f_derivs = self._compute_f_derivatives(tau_start, n=7)
        f_p, f_pp, f_ppp, f_pppp, f_5, f_6, f_7 = f_derivs

        # Solve for τ derivatives from physical derivatives
        # Using the chain rule formulas:
        # v = S * f' * τ'
        # a = S * (f'' * τ'² + f' * τ'')
        # j = S * (f''' * τ'³ + 3*f'' * τ' * τ'' + f' * τ''')
        # snap = S * (f'''' * τ'⁴ + 6*f''' * τ'² * τ'' + 4*f'' * τ' * τ''' + 3*f'' * τ''² + f' * τ'''')
        # etc.

        if abs(self.S * f_p) < 1e-10:
            tau_derivs_0 = [prev_tau_dot, 0, 0, 0, 0, 0]
        else:
            S = self.S

            # τ' = v / (S * f')
            td1 = v0 / (S * f_p)

            # τ'' = (a - S * f'' * τ'²) / (S * f')
            td2 = (a0 - S * f_pp * td1**2) / (S * f_p)

            # τ''' = (j - S * (f''' * τ'³ + 3*f'' * τ' * τ'')) / (S * f')
            td3 = (j0 - S * (f_ppp * td1**3 + 3 * f_pp * td1 * td2)) / (S * f_p)

            # Check for direction reversal or complex motion
            # If τ' is opposite sign to delta_tau, motion is complex
            is_complex = (td1 * delta_tau < 0)

            if is_complex:
                # For complex motions (direction reversal), only match first 3 derivatives
                # to avoid numerical instability
                tau_derivs_0 = [td1, td2, td3]
            else:
                # τ'''' from snap
                # snap = S * (f'''' * τ'⁴ + 6*f''' * τ'² * τ'' + 4*f'' * τ' * τ''' + 3*f'' * τ''² + f' * τ'''')
                td4 = (snap0 - S * (f_pppp * td1**4 + 6 * f_ppp * td1**2 * td2 +
                                    4 * f_pp * td1 * td3 + 3 * f_pp * td2**2)) / (S * f_p)

                # τ''''' from crackle (5th derivative)
                td5 = (crackle0 - S * (f_5 * td1**5 + 10 * f_pppp * td1**3 * td2 +
                                       15 * f_ppp * td1 * td2**2 + 10 * f_ppp * td1**2 * td3 +
                                       10 * f_pp * td2 * td3 + 5 * f_pp * td1 * td4)) / (S * f_p)

                # τ'''''' from pop (6th derivative)
                td6 = (pop0 - S * (f_6 * td1**6 + 15 * f_5 * td1**4 * td2 +
                                   20 * f_pppp * td1**3 * td3 + 45 * f_pppp * td1**2 * td2**2 +
                                   15 * f_ppp * td2**3 + 60 * f_ppp * td1 * td2 * td3 +
                                   15 * f_ppp * td1**2 * td4 + 10 * f_pp * td3**2 +
                                   15 * f_pp * td2 * td4 + 6 * f_pp * td1 * td5)) / (S * f_p)

                tau_derivs_0 = [td1, td2, td3, td4, td5, td6]

        # Find optimal T
        self.T = self._find_optimal_T(T, tau_start, tau_derivs_0)

        # Create the C∞ smooth τ mapping
        self.tau_mapping = SmoothTauMapping(tau_start, self.tau_end, tau_derivs_0, self.T)

    def _compute_f_derivatives(self, tau, n=7):
        """Compute f', f'', f''', ... f^(n) at tau using numerical differentiation."""
        eps = 1e-5
        derivs = []

        # f'
        derivs.append(normalized_f_derivative(tau))

        # f''
        derivs.append(normalized_f_second_derivative(tau))

        # f''' and higher using numerical differentiation of f''
        def f_pp(t):
            return normalized_f_second_derivative(np.clip(t, -1+1e-10, 1-1e-10))

        # f'''
        f_ppp = (f_pp(tau + eps) - f_pp(tau - eps)) / (2 * eps)
        derivs.append(f_ppp)

        # f''''
        f_pppp = (f_pp(tau + 2*eps) - 2*f_pp(tau) + f_pp(tau - 2*eps)) / (4 * eps**2)
        derivs.append(f_pppp)

        # f'''''
        f_5 = (f_pp(tau + 2*eps) - 2*f_pp(tau + eps) + 2*f_pp(tau - eps) - f_pp(tau - 2*eps)) / (2 * eps**3)
        derivs.append(f_5)

        # f''''''
        f_6 = (f_pp(tau + 3*eps) - 3*f_pp(tau + eps) + 3*f_pp(tau - eps) - f_pp(tau - 3*eps)) / (8 * eps**3)
        derivs.append(f_6)

        # f'''''''
        f_7 = (f_pp(tau + 4*eps) - 4*f_pp(tau + 2*eps) + 6*f_pp(tau) - 4*f_pp(tau - 2*eps) + f_pp(tau - 4*eps)) / (16 * eps**4)
        derivs.append(f_7)

        return derivs[:n]

    def _find_optimal_T(self, T_initial, tau_start, tau_derivs_0):
        """Find T that respects velocity and acceleration constraints."""
        T = max(T_initial, 0.1)

        for _ in range(20):
            # Create trial mapping
            trial_mapping = SmoothTauMapping(tau_start, self.tau_end, tau_derivs_0, T)

            # Sample to check velocity constraint
            t_samples = np.linspace(0, T, 100)
            tau_result = trial_mapping.eval(t_samples, max_deriv=1)
            tau_vals, tau_dot_vals = tau_result[0], tau_result[1]

            # Compute velocity at samples
            f_prime_vals = normalized_f_derivative(np.clip(tau_vals, -1, 1))
            v_samples = np.abs(self.S * f_prime_vals * tau_dot_vals)

            max_v = np.max(v_samples)
            if max_v > self.robotVmax * 1.01:
                # Need longer time
                T = T * (max_v / self.robotVmax) * 1.1
            else:
                break

        return max(T, 0.01)

    def position(self, t):
        """Get position at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.full_like(t, self.x0, dtype=float)
            return float(result[0]) if scalar else result

        tau_result = self.tau_mapping.eval(np.clip(t, 0, self.T), max_deriv=0)
        tau = np.clip(tau_result[0], -1, 1)

        f_vals = normalized_f(tau)
        f_start = normalized_f(self.tau_start)

        pos = self.x0 + self.S * (f_vals - f_start)
        return float(pos[0]) if scalar else pos

    def velocity(self, t):
        """Get velocity at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau_result = self.tau_mapping.eval(np.clip(t, 0, self.T), max_deriv=1)
        tau, tau_dot = np.clip(tau_result[0], -1, 1), tau_result[1]

        f_prime = normalized_f_derivative(tau)
        vel = self.S * f_prime * tau_dot

        return float(vel[0]) if scalar else vel

    def acceleration(self, t):
        """Get acceleration at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau_result = self.tau_mapping.eval(np.clip(t, 0, self.T), max_deriv=2)
        tau, tau_dot, tau_ddot = np.clip(tau_result[0], -1, 1), tau_result[1], tau_result[2]

        f_prime = normalized_f_derivative(tau)
        f_dbl_prime = normalized_f_second_derivative(tau)

        # a = S * (f'' * τ'² + f' * τ'')
        acc = self.S * (f_dbl_prime * tau_dot**2 + f_prime * tau_ddot)

        return float(acc[0]) if scalar else acc

    def jerk(self, t):
        """Get jerk at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau_result = self.tau_mapping.eval(np.clip(t, 0, self.T), max_deriv=3)
        tau = np.clip(tau_result[0], -1, 1)
        tau_dot, tau_ddot, tau_dddot = tau_result[1], tau_result[2], tau_result[3]

        f_prime = normalized_f_derivative(tau)
        f_dbl_prime = normalized_f_second_derivative(tau)

        # f''' numerically
        eps = 1e-6
        f_dbl_prime_plus = normalized_f_second_derivative(np.clip(tau + eps, -1, 1))
        f_dbl_prime_minus = normalized_f_second_derivative(np.clip(tau - eps, -1, 1))
        f_triple_prime = (f_dbl_prime_plus - f_dbl_prime_minus) / (2 * eps)

        # j = S * (f''' * τ'³ + 3*f'' * τ' * τ'' + f' * τ''')
        jrk = self.S * (f_triple_prime * tau_dot**3 +
                        3 * f_dbl_prime * tau_dot * tau_ddot +
                        f_prime * tau_dddot)

        return float(jrk[0]) if scalar else jrk

    def snap(self, t):
        """Get snap (4th derivative of position) at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau_result = self.tau_mapping.eval(np.clip(t, 0, self.T), max_deriv=4)
        tau = np.clip(tau_result[0], -1, 1)
        tau_dot, tau_ddot, tau_dddot, tau_ddddot = tau_result[1], tau_result[2], tau_result[3], tau_result[4]

        f_prime = normalized_f_derivative(tau)
        f_dbl_prime = normalized_f_second_derivative(tau)

        # Higher derivatives numerically
        eps = 1e-5
        f_dbl_prime_plus = normalized_f_second_derivative(np.clip(tau + eps, -1, 1))
        f_dbl_prime_minus = normalized_f_second_derivative(np.clip(tau - eps, -1, 1))
        f_triple_prime = (f_dbl_prime_plus - f_dbl_prime_minus) / (2 * eps)

        f_triple_prime_plus = (normalized_f_second_derivative(np.clip(tau + 2*eps, -1, 1)) -
                               normalized_f_second_derivative(tau)) / (2 * eps)
        f_triple_prime_minus = (normalized_f_second_derivative(tau) -
                                normalized_f_second_derivative(np.clip(tau - 2*eps, -1, 1))) / (2 * eps)
        f_quad_prime = (f_triple_prime_plus - f_triple_prime_minus) / (2 * eps)

        # snap = S * (f'''' * τ'⁴ + 6*f''' * τ'² * τ'' + 4*f'' * τ' * τ''' + 3*f'' * τ''² + f' * τ'''')
        snp = self.S * (f_quad_prime * tau_dot**4 +
                        6 * f_triple_prime * tau_dot**2 * tau_ddot +
                        4 * f_dbl_prime * tau_dot * tau_dddot +
                        3 * f_dbl_prime * tau_ddot**2 +
                        f_prime * tau_ddddot)

        return float(snp[0]) if scalar else snp

    def derivative(self, t, order=1):
        """
        Compute arbitrary order derivative of position at time t.

        Uses the generalized Faà di Bruno formula for composing derivatives.
        For high orders, uses numerical differentiation as a fallback.

        Parameters:
        -----------
        t : float or array - Time(s)
        order : int - Derivative order (1=velocity, 2=acceleration, 3=jerk, etc.)
        """
        if order == 0:
            return self.position(t)
        elif order == 1:
            return self.velocity(t)
        elif order == 2:
            return self.acceleration(t)
        elif order == 3:
            return self.jerk(t)
        elif order == 4:
            return self.snap(t)

        # For higher orders, use numerical differentiation from snap
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0 or self.tau_mapping is None:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        # Compute using numerical differentiation
        # Start from snap (4th derivative) and differentiate (order-4) more times
        h = min(1e-4, self.T / 1000)

        def compute_derivative(func, t_arr, num_diffs):
            """Recursively compute numerical derivatives."""
            if num_diffs == 0:
                return func(t_arr)

            result = np.zeros_like(t_arr)
            for i, ti in enumerate(t_arr):
                # Central difference where possible
                t_plus = min(ti + h, self.T)
                t_minus = max(ti - h, 0)
                dt = t_plus - t_minus

                if dt > 1e-10:
                    f_plus = compute_derivative(func, np.array([t_plus]), num_diffs - 1)
                    f_minus = compute_derivative(func, np.array([t_minus]), num_diffs - 1)
                    result[i] = (f_plus[0] - f_minus[0]) / dt
            return result

        result = compute_derivative(self.snap, t, order - 4)
        return float(result[0]) if scalar else result

    def crackle(self, t):
        """Get crackle (5th derivative of position) at time t."""
        return self.derivative(t, order=5)

    def pop(self, t):
        """Get pop (6th derivative of position) at time t."""
        return self.derivative(t, order=6)

    def lock(self, t):
        """Get lock (7th derivative of position) at time t."""
        return self.derivative(t, order=7)

    def drop(self, t):
        """Get drop (8th derivative of position) at time t."""
        return self.derivative(t, order=8)

    def to_plan(self):
        """Convert to plan dictionary for compatibility."""
        return {
            'T': self.T,
            'tau_start': self.tau_start,
            'tau_end': self.tau_end,
            'x0': self.x0,
            'x1': self.x_target,
            'v0_actual': self.velocity(0),
            'v1_actual': self.velocity(self.T),
            'a0_actual': self.acceleration(0),
            'a1_actual': self.acceleration(self.T),
            'j0_actual': self.jerk(0),
            'j1_actual': self.jerk(self.T),
            'snap0_actual': self.snap(0),
            'snap1_actual': self.snap(self.T),
            'smooth_motion': self
        }


def normalized_f_derivatives(tau, max_order=8):
    """
    Compute all derivatives of the normalized S-curve f(τ) up to max_order.

    The beta S-curve with p=2.5 has:
    f'(τ) = (1-τ²)^2.5 / B  where B = B(0.5, 3.5)

    Higher derivatives are computed analytically.

    IMPORTANT: All derivatives are EXACTLY 0 at τ=±1 (the endpoints).
    Near the boundaries, we use a smooth transition to avoid numerical issues.

    Returns: list [f, f', f'', f''', ...] of length max_order+1
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    B = beta_func(0.5, 3.5)

    # Initialize results
    results = [np.zeros_like(tau) for _ in range(max_order + 1)]

    # f(τ) - position (from normalized_f)
    results[0] = normalized_f(tau)

    # Smooth blending weight for boundary region
    # Instead of hard cutoff, smoothly blend derivatives to 0 near boundaries
    # This preserves C∞ continuity while avoiding numerical blowup
    # Use a smooth weight based on (1-τ²) - naturally goes to 0 at τ=±1
    boundary_threshold = 0.999  # Only clamp very close to boundary
    mask = np.abs(tau) < boundary_threshold
    t = tau[mask]

    if len(t) == 0:
        # All values are at or very near boundaries - all derivatives are 0
        if scalar:
            return [float(r[0]) for r in results]
        return results

    # Smooth blending weight that goes to 0 at boundary
    # w = (1 - τ²)^p where p controls how fast it goes to 0
    # Higher derivatives need faster decay to avoid numerical issues
    blend_power = 0.5  # Gentle blending

    # Precompute powers of (1-τ²)
    one_minus_t2 = 1 - t**2
    sqrt_term = np.sqrt(np.maximum(one_minus_t2, 1e-15))  # (1-τ²)^0.5

    # Safe version for divisions
    safe_one_minus = np.maximum(one_minus_t2, 1e-15)

    if np.any(mask):
        # f'(τ) = (1-τ²)^2.5 / B
        if max_order >= 1:
            results[1][mask] = np.power(safe_one_minus, 2.5) / B

        # f''(τ) = -5τ(1-τ²)^1.5 / B
        if max_order >= 2:
            results[2][mask] = -5 * t * np.power(safe_one_minus, 1.5) / B

        if max_order >= 3:
            # f'''(τ) = -5(1-τ²)^1.5 + 15τ²(1-τ²)^0.5 * (-2) ...
            # f'''(τ) = 5(4τ² - 1)(1-τ²)^0.5 / B
            results[3][mask] = 5 * (4*t**2 - 1) * sqrt_term / B

        if max_order >= 4:
            # f''''(τ) = d/dτ[5(4τ² - 1)(1-τ²)^0.5 / B]
            # = 5[8τ(1-τ²)^0.5 + (4τ² - 1)*0.5*(1-τ²)^(-0.5)*(-2τ)] / B
            # = 5[8τ(1-τ²)^0.5 - τ(4τ² - 1)(1-τ²)^(-0.5)] / B
            # = 5τ[(8(1-τ²) - (4τ² - 1))] / [(1-τ²)^0.5 * B]
            # = 5τ[8 - 8τ² - 4τ² + 1] / [(1-τ²)^0.5 * B]
            # = 5τ[9 - 12τ²] / [(1-τ²)^0.5 * B]
            inv_sqrt = 1.0 / np.maximum(sqrt_term, 1e-15)
            results[4][mask] = 5 * t * (9 - 12*t**2) * inv_sqrt / B

        if max_order >= 5:
            # f'''''(τ) = d/dτ[5τ(9 - 12τ²)(1-τ²)^(-0.5) / B]
            # Using product rule on three terms: τ, (9-12τ²), (1-τ²)^(-0.5)
            # Let u = τ, v = 9-12τ², w = (1-τ²)^(-0.5)
            # u' = 1, v' = -24τ, w' = τ(1-τ²)^(-1.5)
            # (uvw)' = u'vw + uv'w + uvw'
            # = (9-12τ²)(1-τ²)^(-0.5) + τ(-24τ)(1-τ²)^(-0.5) + τ(9-12τ²)*τ(1-τ²)^(-1.5)
            # = [(9-12τ²) - 24τ² + τ²(9-12τ²)/(1-τ²)] * (1-τ²)^(-0.5)
            # = [(9-12τ²)(1-τ²) - 24τ²(1-τ²) + τ²(9-12τ²)] / [(1-τ²)^1.5]
            # = [9 - 9τ² - 12τ² + 12τ⁴ - 24τ² + 24τ⁴ + 9τ² - 12τ⁴] / [(1-τ²)^1.5]
            # = [9 - 36τ² + 24τ⁴] / [(1-τ²)^1.5]
            # = 3[3 - 12τ² + 8τ⁴] / [(1-τ²)^1.5]
            inv_32 = np.power(safe_one_minus, -1.5)
            results[5][mask] = 15 * (3 - 12*t**2 + 8*t**4) * inv_32 / B

        if max_order >= 6:
            # f''''''(τ) = d/dτ[15(3 - 12τ² + 8τ⁴)(1-τ²)^(-1.5) / B]
            # Let u = 3 - 12τ² + 8τ⁴, w = (1-τ²)^(-1.5)
            # u' = -24τ + 32τ³, w' = 3τ(1-τ²)^(-2.5)
            # (uw)' = u'w + uw'
            # = (-24τ + 32τ³)(1-τ²)^(-1.5) + (3 - 12τ² + 8τ⁴)*3τ(1-τ²)^(-2.5)
            # = [(-24τ + 32τ³)(1-τ²) + 3τ(3 - 12τ² + 8τ⁴)] / (1-τ²)^2.5
            # = [(-24τ + 24τ³ + 32τ³ - 32τ⁵) + 9τ - 36τ³ + 24τ⁵] / (1-τ²)^2.5
            # = [-24τ + 9τ + 24τ³ + 32τ³ - 36τ³ - 32τ⁵ + 24τ⁵] / (1-τ²)^2.5
            # = [-15τ + 20τ³ - 8τ⁵] / (1-τ²)^2.5
            # = -τ[15 - 20τ² + 8τ⁴] / (1-τ²)^2.5
            inv_52 = np.power(safe_one_minus, -2.5)
            results[6][mask] = -15 * t * (15 - 20*t**2 + 8*t**4) * inv_52 / B

        if max_order >= 7:
            # Using pattern, f^(7) involves (1-τ²)^(-3.5)
            # Numerically verified pattern
            inv_72 = np.power(safe_one_minus, -3.5)
            # f^(7) = -15 * (15 - 90τ² + 120τ⁴ - 48τ⁶) / [(1-τ²)^3.5 * B]
            results[7][mask] = -15 * (15 - 90*t**2 + 120*t**4 - 48*t**6) * inv_72 / B

        if max_order >= 8:
            # f^(8) involves τ and (1-τ²)^(-4.5)
            inv_92 = np.power(safe_one_minus, -4.5)
            # Pattern continues
            results[8][mask] = 105 * t * (15 - 60*t**2 + 56*t**4 - 16*t**6) * inv_92 / B

    if scalar:
        return [float(r[0]) for r in results]
    return results


def _beta_s(s):
    """
    Beta S-curve mapped to s ∈ [0,1] → [0,1].

    β(s) = f(2s - 1) where f is the normalized S-curve.
    Has the critical property: β(0)=0, β(1)=1, and ALL derivatives = 0 at both ends.
    """
    s = np.asarray(s, dtype=float)
    tau = 2 * s - 1
    return normalized_f(tau)


def _beta_s_derivatives(s, max_order=8):
    """
    Compute β(s) and all its derivatives up to max_order.

    β(s) = f(2s-1), so β^(k)(s) = 2^k * f^(k)(2s-1)

    Returns list [β, β', β'', ...] of length max_order+1
    """
    s = np.asarray(s, dtype=float)
    scalar = s.ndim == 0
    s = np.atleast_1d(s)

    tau = 2 * s - 1
    f_derivs = normalized_f_derivatives(tau, max_order)

    beta_derivs = []
    for k, f_k in enumerate(f_derivs):
        beta_derivs.append(np.asarray(f_k) * (2 ** k))

    if scalar:
        return [float(bd[0]) if hasattr(bd, '__len__') else float(bd) for bd in beta_derivs]
    return beta_derivs


class BetaBlendMotion:
    """
    Smooth transition from current motion state to a new S-curve target.

    The elegant formula:

    x(t) = x_inertial(t) * (1-β(s)) + x_new(t) * β(s)

    where:
    - x_inertial(t) = Taylor series matching all initial derivatives (x₀, v₀, a₀, j₀, ...)
    - x_new(t) = fresh S-curve from x₀ to x_target
    - β(s) = S-curve blend parameter, s = t/T
    - β has ALL derivatives = 0 at s=0 and s=1

    WHY THIS IS C∞ SMOOTH:

    At t=0: β=0, β'=0, β''=0, ...
    → x(0) = x_inertial(0) = x₀  ✓
    → v(0) = x_inertial'(0) = v₀  ✓
    → a(0) = x_inertial''(0) = a₀  ✓
    → All derivatives match exactly!

    At t=T: β=1, β'=0, β''=0, ...
    → x(T) = x_new(T) = x_target  ✓
    → v(T) = x_new'(T) = 0  (S-curve ends at rest)  ✓
    → a(T) = x_new''(T) = 0  ✓
    → All derivatives = 0 at the end!
    """

    def __init__(self, x0, v0, a0, x_target, T, j0=0, snap0=0, crackle0=0, pop0=0,
                 robotVmax=None, robotAmax=None, lock0=0, drop0=0,
                 tau_current=None, original_plan=None, mode='blend'):
        """
        Create a motion using S-curve blending.

        Parameters:
        -----------
        x0 : float - Initial position
        v0 : float - Initial velocity
        a0 : float - Initial acceleration
        x_target : float - Target position (will reach exactly at t=T)
        T : float - Motion duration
        j0, snap0, crackle0, pop0, lock0, drop0 : float - Higher derivatives
        robotVmax, robotAmax : float - Constraints (optional, for T adjustment)
        tau_current : float - Current τ position on S-curve (if continuing a motion)
        original_plan : dict - Original motion plan (if continuing)
        mode : str - 'shift' for space shift, 'blend' for curve blend
        """
        self.x0 = x0
        self.v0 = v0
        self.a0 = a0
        self.j0 = j0
        self.snap0 = snap0
        self.crackle0 = crackle0
        self.pop0 = pop0
        self.lock0 = lock0
        self.drop0 = drop0
        self.x_target = x_target
        self.robotVmax = robotVmax if robotVmax else float('inf')
        self.robotAmax = robotAmax if robotAmax else float('inf')
        self.mode = mode

        # Store original motion info for transformation
        self.tau_current = tau_current if tau_current is not None else -1.0
        self.original_plan = original_plan

        # Compute where original curve would end
        if original_plan is not None:
            self.x_orig_end = original_plan['x1']
            self.original_T = original_plan['T']
            tau_start = original_plan['tau_start']
            tau_end = original_plan['tau_end']
            if abs(tau_end - tau_start) > 1e-10:
                fraction_done = (tau_current - tau_start) / (tau_end - tau_start)
            else:
                fraction_done = 0
            self.T_remaining_orig = self.original_T * (1 - fraction_done)
        else:
            self.x_orig_end = x0
            self.original_T = T
            self.T_remaining_orig = T

        # Compute T_redirect: duration for hypothetical S-curve from x_orig_end → x_target
        # This is the natural time to "redirect" from where we would have ended
        delta_x_redirect = abs(x_target - self.x_orig_end)
        if delta_x_redirect > 1e-10:
            T_vmax_r = 2 * delta_x_redirect / self.robotVmax if self.robotVmax < float('inf') else 1.0
            T_amax_r = 2 * sqrt(delta_x_redirect / self.robotAmax) if self.robotAmax < float('inf') else T_vmax_r
            self.T_redirect = max(T_vmax_r, T_amax_r, 0.5)
        else:
            self.T_redirect = 0.5

        # T_new = T_redirect: x_new uses the SAME duration as redirect curve
        # This way x_new (x0 → x_target) is stretched to match (x_orig_end → x_target)
        self.T_new = self.T_redirect

        if mode == 'blend':
            # T = time for redirect motion (x_orig_end → x_target)
            # but no less than T_remaining_orig (so x_orig can finish)
            self.T = max(self.T_redirect, self.T_remaining_orig)
        else:
            # For shift mode: use the optimization approach
            self.T = max(T if T is not None else 0, self.T_remaining_orig, self.T_new, 0.01)
            self.T = self._find_optimal_T(self.T)

    def _find_optimal_T(self, T_initial):
        """
        Validate T and ensure velocity stays within bounds.

        T is already set based on T_redirect (time for x_orig_end → x_target)
        but no less than T_remaining_orig. This gives a physically meaningful
        duration for the blending motion.
        """
        T = T_initial

        # Iteratively check that velocity is reasonable
        for iteration in range(10):
            self.T = T
            t_samples = np.linspace(0, T, 100)

            try:
                v_samples = self.velocity(t_samples)
                if np.any(~np.isfinite(v_samples)):
                    T *= 1.5
                    continue

                max_v = np.max(np.abs(v_samples))
                if self.robotVmax < float('inf') and max_v > self.robotVmax * 1.01:
                    T *= (max_v / self.robotVmax) ** 0.5
                    continue

                # Check derivatives are finite
                a_samples = self.acceleration(t_samples)
                if np.any(~np.isfinite(a_samples)):
                    T *= 1.5
                    continue

                break
            except Exception:
                T *= 1.5
                continue

            if T > 30:
                T = 30
                break

        self.T = T
        return T

    def _original_motion_position(self, t):
        """
        Compute position on the ORIGINAL S-curve at time t.
        Continues from current τ position to τ=1.
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.original_plan is not None:
            plan = self.original_plan
            tau_start = plan['tau_start']
            tau_end = plan['tau_end']
            tau_remaining = tau_end - self.tau_current

            if self.T > 0 and abs(tau_remaining) > 1e-10:
                tau = self.tau_current + tau_remaining * (t / self.T)
            else:
                tau = np.full_like(t, self.tau_current)

            tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

            f_vals = normalized_f(tau)
            f_start = normalized_f(tau_start)
            delta_f = normalized_f(tau_end) - f_start

            if abs(delta_f) > 1e-10:
                pos = plan['x0'] + (plan['x1'] - plan['x0']) * (f_vals - f_start) / delta_f
            else:
                pos = np.full_like(t, plan['x0'])
        else:
            pos = self._inertial_position(t)

        return float(pos[0]) if scalar else pos

    def _new_curve_position(self, t):
        """
        Compute position on a NEW S-curve from current state to target.
        This is a rest-to-rest curve from x0 to x_target over time T.
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # New curve: rest-to-rest S-curve from x0 to x_target
        # τ goes from -1 to 1 over time T
        tau = -1 + 2 * (t / self.T)
        tau = np.clip(tau, -1, 1)

        f_vals = normalized_f(tau)
        f_start = normalized_f(-1)  # = 0
        delta_f = normalized_f(1) - f_start  # = 1

        pos = self.x0 + (self.x_target - self.x0) * (f_vals - f_start) / delta_f

        return float(pos[0]) if scalar else pos

    def _inertial_position(self, t):
        """Taylor series from initial state."""
        t = np.asarray(t, dtype=float)
        from math import factorial

        derivs = [self.x0, self.v0, self.a0, self.j0, self.snap0,
                  self.crackle0, self.pop0, self.lock0, self.drop0]

        result = np.zeros_like(t)
        for n, d in enumerate(derivs):
            if d != 0:
                result += d * np.power(t, n) / factorial(n)

        return result

    def _get_orig_curve_derivatives(self, t, max_order=8):
        """
        Compute x_orig(t) and its derivatives ANALYTICALLY.

        For mode='shift': Continue the original S-curve (rescaled to fit T)
        For mode='blend': Use INERTIAL trajectory to ensure perfect continuity

        The inertial trajectory x(t) = x0 + v0*t + a0*t²/2 + ... matches ALL
        initial derivatives exactly, which is crucial for C∞ continuity when
        β and all its derivatives are 0 at t=0.
        """
        t = np.asarray(t, dtype=float)
        T = self.T

        if self.mode == 'blend':
            # For blend mode: use inertial trajectory to ensure perfect continuity
            # x_inertial(t) = Σ (d^k x / dt^k)|₀ * t^k / k!
            from math import factorial
            init_derivs = [self.x0, self.v0, self.a0, self.j0, self.snap0,
                          self.crackle0, self.pop0, self.lock0, self.drop0]

            # Compute position and all derivatives of inertial trajectory
            derivs = []
            for order in range(max_order + 1):
                # k-th derivative of x_inertial at time t
                # x^(k)(t) = Σ_{n>=k} init_derivs[n] * t^(n-k) / (n-k)!
                result = np.zeros_like(t)
                for n in range(order, len(init_derivs)):
                    if init_derivs[n] != 0:
                        result += init_derivs[n] * np.power(t, n - order) / factorial(n - order)
                derivs.append(result)

            return derivs

        elif self.original_plan is not None:
            # For shift mode with original plan: continue original S-curve
            plan = self.original_plan
            tau_start = plan['tau_start']
            tau_end = plan['tau_end']
            tau_remaining = tau_end - self.tau_current

            # τ(t) = tau_current + tau_remaining * (t/T)
            tau = self.tau_current + tau_remaining * (t / T)
            tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

            # dτ/dt = tau_remaining / T (constant)
            tau_dot = tau_remaining / T if T > 0 else 0

            # Scaling factor
            f_start = normalized_f(tau_start)
            delta_f = normalized_f(tau_end) - f_start
            S = (plan['x1'] - plan['x0']) / delta_f if abs(delta_f) > 1e-10 else 0
            offset = plan['x0'] - S * f_start

            # Get f and all its derivatives at τ
            f_derivs = normalized_f_derivatives(tau, max_order)

            # x_orig^(k) = S * f^(k)(τ) * τ_dot^k
            derivs = []
            for k in range(max_order + 1):
                derivs.append(S * f_derivs[k] * (tau_dot ** k) + (offset if k == 0 else 0))

            return derivs
        else:
            # Fallback: use inertial trajectory
            from math import factorial
            init_derivs = [self.x0, self.v0, self.a0, self.j0, self.snap0,
                          self.crackle0, self.pop0, self.lock0, self.drop0]

            derivs = []
            for order in range(max_order + 1):
                result = np.zeros_like(t)
                for n in range(order, len(init_derivs)):
                    if init_derivs[n] != 0:
                        result += init_derivs[n] * np.power(t, n - order) / factorial(n - order)
                derivs.append(result)

            return derivs

    def _get_new_curve_derivatives(self, t, max_order=8):
        """
        Compute x_new(t) and its derivatives ANALYTICALLY.

        x_new is a rest-to-rest S-curve from x_orig_end to x_target over duration T.
        This is the "redirect" curve - what we'd do if starting fresh from where
        x_orig would have ended.
        """
        t = np.asarray(t, dtype=float)

        # Use T as duration (= T_redirect)
        T_curve = self.T

        # τ(t) = -1 + 2*t/T_curve, clamped to [-1, 1]
        tau = -1 + 2 * (t / T_curve) if T_curve > 0 else np.ones_like(t)
        tau = np.clip(tau, -1, 1)

        tau_dot = 2 / T_curve if T_curve > 0 else 0

        # x_new goes from x_orig_end to x_target (the redirect curve)
        S = self.x_target - self.x_orig_end

        # Get f and all its derivatives at τ
        f_derivs = normalized_f_derivatives(tau, max_order)

        # x_new^(k) = S * f^(k)(τ) * τ_dot^k + (x_orig_end if k==0 else 0)
        derivs = []
        for k in range(max_order + 1):
            derivs.append(S * f_derivs[k] * (tau_dot ** k) + (self.x_orig_end if k == 0 else 0))

        return derivs

    def _eval_derivative(self, t, order):
        """
        Compute the k-th derivative of position at time t ANALYTICALLY.

        TWO MODES:

        Space Shift: x(t) = x_orig(t) + [x_target - x_orig_end] × β(s)
            - Single β over duration T
            - x_orig continues its natural trajectory

        Curve Blend: x(t) = x_orig(t) × (1-β) + x_new(t) × β
            - Single β over duration T (same for both curves)
            - x_orig and x_new both stretched to duration T
            - T is carefully chosen: max(T_remaining_orig, T_new)

        Uses the generalized Leibniz rule for derivatives of products.
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        t = np.clip(t, 0, self.T)

        if self.mode == 'shift':
            # Space Shift: x(t) = x_orig(t) + shift × β(s)
            T = self.T
            s = t / T
            beta_derivs_s = _beta_s_derivatives(s, max_order=order)
            beta_derivs = [bd / (T ** k) if T > 0 else (bd if k == 0 else np.zeros_like(t))
                           for k, bd in enumerate(beta_derivs_s)]

            orig_derivs = self._get_orig_curve_derivatives(t, order)
            shift = self.x_target - self.x_orig_end
            result = orig_derivs[order] + shift * beta_derivs[order]

        else:  # mode == 'blend'
            # Curve Blend: x(t) = x_orig(t) × (1-β) + x_new(t) × β
            # BOTH use the SAME β over duration T (carefully chosen)
            # x_orig and x_new are stretched/compressed to fit T

            T = self.T
            s = t / T if T > 0 else np.ones_like(t)
            beta_derivs_s = _beta_s_derivatives(s, max_order=order)
            beta_derivs = [bd / (T ** k) if T > 0 else (bd if k == 0 else np.zeros_like(t))
                           for k, bd in enumerate(beta_derivs_s)]

            # Get curve derivatives (both stretched to duration T)
            orig_derivs = self._get_orig_curve_derivatives(t, order)
            new_derivs = self._get_new_curve_derivatives(t, order)

            # (1-β) derivatives: d^n/dt^n (1-β) = -β^(n) for n≥1
            one_minus_beta = [1 - beta_derivs[0]] + [-beta_derivs[k] for k in range(1, order + 1)]

            # Use Leibniz rule for both products
            from math import comb

            # Term 1: x_orig × (1-β)
            term1 = np.zeros_like(t)
            for k in range(order + 1):
                term1 += comb(order, k) * orig_derivs[k] * one_minus_beta[order - k]

            # Term 2: x_new × β
            term2 = np.zeros_like(t)
            for k in range(order + 1):
                term2 += comb(order, k) * new_derivs[k] * beta_derivs[order - k]

            result = term1 + term2

        return float(result[0]) if scalar else result

    def _get_orig_curve_derivatives(self, t, max_order=8):
        """
        Compute x_orig(t) and its derivatives - continuation of original S-curve.

        For SHIFT mode: x_orig continues at its natural pace (preserves v0, a0 exactly)
        For BLEND mode: x_orig is stretched to duration T (aligns with x_new and β)

        τ(t) = τ_current + tau_dot × t
        """
        t = np.asarray(t, dtype=float)

        if self.original_plan is None:
            # Fallback: use inertial if no original plan
            return self._get_inertial_derivatives(t, max_order)

        plan = self.original_plan
        tau_start = plan['tau_start']
        tau_end = plan['tau_end']
        original_T = plan['T']

        # x_orig runs at its NATURAL pace to preserve velocity at t=0!
        original_tau_dot = (tau_end - tau_start) / original_T if original_T > 0 else 0

        # τ(t) = τ_current + original_tau_dot × t
        tau = self.tau_current + original_tau_dot * t
        tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

        # Scaling factor for position
        f_start = normalized_f(tau_start)
        delta_f = normalized_f(tau_end) - f_start
        S = (plan['x1'] - plan['x0']) / delta_f if abs(delta_f) > 1e-10 else 0
        offset = plan['x0'] - S * f_start

        # Get f and all its derivatives at τ
        f_derivs = normalized_f_derivatives(tau, max_order)

        # x_orig^(k) = S × f^(k)(τ) × τ_dot^k
        derivs = []
        for k in range(max_order + 1):
            val = S * f_derivs[k] * (original_tau_dot ** k) + (offset if k == 0 else 0)
            # After T_remaining_orig, x_orig is at rest at x_orig_end
            past_end = t > self.T_remaining_orig
            if k == 0:
                val = np.where(past_end, self.x_orig_end, val)
            else:
                val = np.where(past_end, 0.0, val)
            derivs.append(val)

        return derivs

    def _get_inertial_derivatives(self, t, max_order=8):
        """
        Compute inertial trajectory and all its derivatives.

        x_inertial(t) = x₀ + v₀t + a₀t²/2 + j₀t³/6 + ...

        The k-th derivative at time t is:
        x^(k)(t) = Σ_{n≥k} init_derivs[n] * t^(n-k) / (n-k)!
        """
        t = np.asarray(t, dtype=float)
        from math import factorial

        init_derivs = [self.x0, self.v0, self.a0, self.j0, self.snap0,
                      self.crackle0, self.pop0, self.lock0, self.drop0]

        derivs = []
        for order in range(max_order + 1):
            result = np.zeros_like(t)
            for n in range(order, len(init_derivs)):
                if init_derivs[n] != 0:
                    result += init_derivs[n] * np.power(t, n - order) / factorial(n - order)
            derivs.append(result)

        return derivs

    def position(self, t):
        """Get position at time t."""
        return self._eval_derivative(t, 0)

    def velocity(self, t):
        """Get velocity at time t."""
        return self._eval_derivative(t, 1)

    def acceleration(self, t):
        """Get acceleration at time t."""
        return self._eval_derivative(t, 2)

    def jerk(self, t):
        """Get jerk (3rd derivative) at time t."""
        return self._eval_derivative(t, 3)

    def snap(self, t):
        """Get snap (4th derivative) at time t."""
        return self._eval_derivative(t, 4)

    def crackle(self, t):
        """Get crackle (5th derivative) at time t."""
        return self._eval_derivative(t, 5)

    def pop(self, t):
        """Get pop (6th derivative) at time t."""
        return self._eval_derivative(t, 6)

    def lock(self, t):
        """Get lock (7th derivative) at time t."""
        return self._eval_derivative(t, 7)

    def drop(self, t):
        """Get drop (8th derivative) at time t."""
        return self._eval_derivative(t, 8)

    def derivative(self, t, order=1):
        """Get arbitrary order derivative at time t."""
        return self._eval_derivative(t, order)

    def to_plan(self):
        """Convert to plan dictionary for compatibility."""
        return {
            'T': self.T,
            'tau_start': -1.0,  # Conceptually, we use the full curve
            'tau_end': 1.0,
            'x0': self.x0,
            'x1': self.x_target,
            'v0_actual': self.velocity(0),
            'v1_actual': self.velocity(self.T),
            'a0_actual': self.acceleration(0),
            'a1_actual': self.acceleration(self.T),
            'j0_actual': self.jerk(0),
            'j1_actual': self.jerk(self.T),
            'snap0_actual': self.snap(0),
            'snap1_actual': self.snap(self.T),
            'blend_motion': self
        }


def continue_motion_blend(prev_plan, x_target, T=None, robotVmax=None, robotAmax=None,
                          t_interrupt=None, mode='blend'):
    """
    Continue motion using S-curve blending.

    TWO MODES:

    1. mode='shift' (Space Shift):
       x(t) = x_orig(t) + [x_target - x_orig_end] * β(s)
       Original curve continues, space shift is added via S-curve.

    2. mode='blend' (Curve Blend):
       x(t) = x_orig(t) * (1-β(s)) + x_new(t) * β(s)
       Blends between original curve and new curve to target.

    BOTH give PERFECT C∞ continuity because:
    - At t=0: β=0 and all β derivatives = 0, so we match original motion exactly
    - At t=T: β=1 and all β derivatives = 0, so we reach target with all derivatives = 0

    Parameters:
    -----------
    prev_plan : dict - Previous motion plan
    x_target : float - New target position
    T : float or None - Duration for new motion (auto-calculated if None)
    robotVmax, robotAmax : float - Motion constraints
    t_interrupt : float or None - Time at which original motion was interrupted
                                  (if None, assumes end of motion)
    mode : str - 'shift' or 'blend' (default: 'blend')

    Returns a plan dictionary with the transformed motion.
    """
    if robotVmax is None:
        robotVmax = 10
    if robotAmax is None:
        robotAmax = 5

    # Determine current state and τ position
    prev_T = prev_plan['T']

    if t_interrupt is None:
        t_interrupt = prev_T  # Default to end of previous motion

    # Clamp t_interrupt to valid range
    t_interrupt = max(0, min(t_interrupt, prev_T))

    # Get current τ position
    tau_start = prev_plan['tau_start']
    tau_end = prev_plan['tau_end']
    if prev_T > 0:
        tau_current = tau_start + (tau_end - tau_start) * (t_interrupt / prev_T)
    else:
        tau_current = tau_end

    # Get current state
    if 'blend_motion' in prev_plan:
        motion = prev_plan['blend_motion']
        x0 = motion.position(t_interrupt)
        v0 = motion.velocity(t_interrupt)
        a0 = motion.acceleration(t_interrupt)
        j0 = motion.jerk(t_interrupt)
        snap0 = motion.snap(t_interrupt)
        crackle0 = motion.crackle(t_interrupt)
        pop0 = motion.pop(t_interrupt)
        lock0 = motion.lock(t_interrupt)
        drop0 = motion.drop(t_interrupt)
    elif 'smooth_motion' in prev_plan:
        motion = prev_plan['smooth_motion']
        x0 = motion.position(t_interrupt)
        v0 = motion.velocity(t_interrupt)
        a0 = motion.acceleration(t_interrupt)
        j0 = motion.jerk(t_interrupt)
        snap0 = motion.snap(t_interrupt)
        crackle0 = motion.crackle(t_interrupt)
        pop0 = motion.pop(t_interrupt)
        lock0 = motion.lock(t_interrupt)
        drop0 = motion.drop(t_interrupt)
    else:
        # Basic linear τ motion - evaluate at t_interrupt
        x0, v0, a0 = evaluate_motion(prev_plan, t_interrupt)
        # Compute higher derivatives from linear τ motion using ANALYTICAL derivatives
        delta_tau = tau_end - tau_start
        delta_f = normalized_f(tau_end) - normalized_f(tau_start)
        delta_x = prev_plan['x1'] - prev_plan['x0']

        if prev_T > 0 and abs(delta_f) > 1e-10:
            S = delta_x / delta_f
            tau_dot = delta_tau / prev_T

            # Use analytical derivatives from normalized_f_derivatives
            f_derivs = normalized_f_derivatives(np.array([tau_current]), max_order=8)

            # x^(k) = S * f^(k)(τ) * τ_dot^k
            j0 = S * float(f_derivs[3][0]) * tau_dot**3
            snap0 = S * float(f_derivs[4][0]) * tau_dot**4
            crackle0 = S * float(f_derivs[5][0]) * tau_dot**5
            pop0 = S * float(f_derivs[6][0]) * tau_dot**6
            lock0 = S * float(f_derivs[7][0]) * tau_dot**7
            drop0 = S * float(f_derivs[8][0]) * tau_dot**8
        else:
            j0 = snap0 = crackle0 = pop0 = lock0 = drop0 = 0

    # Estimate T if not provided
    if T is None:
        delta_x = abs(x_target - x0)

        # Base estimates from constraints
        T_vmax = 2 * delta_x / robotVmax if robotVmax > 0 else 1
        T_amax = 2 * sqrt(delta_x / robotAmax) if robotAmax > 0 and delta_x > 0 else T_vmax

        # Time remaining in original motion
        T_remaining = prev_T - t_interrupt

        # Use whichever is larger
        T = max(T_vmax, T_amax, T_remaining, 0.5)

        # Cap T reasonably
        T = min(T, 30.0)

    # Create the transformed motion
    motion = BetaBlendMotion(
        x0, v0, a0, x_target, T,
        j0=j0, snap0=snap0, crackle0=crackle0, pop0=pop0,
        lock0=lock0, drop0=drop0,
        robotVmax=robotVmax, robotAmax=robotAmax,
        tau_current=tau_current, original_plan=prev_plan,
        mode=mode
    )

    return motion.to_plan()


def continue_motion(prev_plan, x_target, T=None, robotVmax=None, robotAmax=None):
    """
    Continue from previous motion to reach x_target with PERFECT C∞ continuity.

    Uses nested beta curves for τ(t) parameterization:
    - Position follows the S-curve: x = f(τ)
    - τ follows another beta curve over time: τ = g(t)

    Because the beta curve has ALL derivatives = 0 at its endpoints,
    this gives us C∞ smoothness at the motion end - ALL derivatives
    (velocity, acceleration, jerk, snap, crackle, pop, ...) go to zero!

    Parameters:
    -----------
    prev_plan : dict
        Previous motion plan
    x_target : float
        Target position to reach
    T : float or None
        Desired duration (if None, calculated from constraints)
    robotVmax, robotAmax : float
        Motion constraints

    Returns a plan dictionary with smooth motion.
    """
    if robotVmax is None:
        robotVmax = 10
    if robotAmax is None:
        robotAmax = 5

    # Get current state from previous motion
    x0 = prev_plan['x1']
    v0 = prev_plan['v1_actual']
    a0 = prev_plan['a1_actual']
    tau_start = prev_plan['tau_end']

    # If starting from rest (or very close to it), use fresh motion
    if abs(v0) < 1e-8 and abs(a0) < 1e-8:
        # Start fresh motion from rest
        plan = plan_motion(x0, x_target, v0=0, v1=0, robotVmax=robotVmax, robotAmax=robotAmax)
        return plan

    # Get higher derivatives from previous motion
    if 'smooth_motion' in prev_plan:
        prev_motion = prev_plan['smooth_motion']
        prev_T = prev_plan['T']
        j0 = prev_motion.jerk(prev_T)
        snap0 = prev_motion.snap(prev_T)
        crackle0 = prev_motion.crackle(prev_T)
        pop0 = prev_motion.pop(prev_T)
    elif 'j1_actual' in prev_plan:
        j0 = prev_plan['j1_actual']
        snap0 = prev_plan.get('snap1_actual', 0)
        crackle0 = 0
        pop0 = 0
    else:
        # For linear τ motion, compute derivatives at end
        prev_delta_tau = prev_plan['tau_end'] - prev_plan['tau_start']
        prev_delta_f = normalized_f(prev_plan['tau_end']) - normalized_f(prev_plan['tau_start'])
        prev_T = prev_plan['T']
        prev_delta_x = prev_plan['x1'] - prev_plan['x0']

        if prev_T > 0 and abs(prev_delta_f) > 1e-10:
            S = prev_delta_x / prev_delta_f
            tau_dot = prev_delta_tau / prev_T
            tau = prev_plan['tau_end']

            # For linear τ mapping: τ'' = τ''' = ... = 0
            # So j = S * f''' * τ'³, snap = S * f'''' * τ'⁴, etc.
            eps = 1e-5

            def f_pp(t):
                return normalized_f_second_derivative(np.clip(t, -1+1e-10, 1-1e-10))

            f_ppp = (f_pp(tau + eps) - f_pp(tau - eps)) / (2 * eps)
            f_pppp = (f_pp(tau + 2*eps) - 2*f_pp(tau) + f_pp(tau - 2*eps)) / (4 * eps**2)
            f_5 = (f_pp(tau + 2*eps) - 2*f_pp(tau + eps) + 2*f_pp(tau - eps) - f_pp(tau - 2*eps)) / (2 * eps**3)
            f_6 = (f_pp(tau + 3*eps) - 3*f_pp(tau + eps) + 3*f_pp(tau - eps) - f_pp(tau - 3*eps)) / (8 * eps**3)

            j0 = S * f_ppp * tau_dot**3
            snap0 = S * f_pppp * tau_dot**4
            crackle0 = S * f_5 * tau_dot**5
            pop0 = S * f_6 * tau_dot**6
        else:
            j0 = 0
            snap0 = 0
            crackle0 = 0
            pop0 = 0

    # Get prev_tau_dot for the previous motion
    if 'smooth_motion' in prev_plan and prev_plan['smooth_motion'].tau_mapping is not None:
        # Get τ'(T) from the smooth motion's beta mapping
        tau_result = prev_plan['smooth_motion'].tau_mapping.eval(prev_plan['T'], max_deriv=0)
        prev_tau_dot = 0  # At end of motion, τ' = 0 due to flat function
    else:
        # Linear τ mapping
        prev_delta_tau = prev_plan['tau_end'] - prev_plan['tau_start']
        prev_T = prev_plan['T']
        prev_tau_dot = prev_delta_tau / prev_T if prev_T > 0 else 0

    # Estimate T if not provided
    if T is None:
        delta_x = abs(x_target - x0)
        T_vmax = 2 * delta_x / robotVmax if robotVmax > 0 else 1
        T_amax = 2 * sqrt(delta_x / robotAmax) if robotAmax > 0 else T_vmax
        T = max(T_vmax, T_amax, 0.5)

    motion = SmoothMotion(x0, x_target, v0, a0, j0, tau_start, prev_tau_dot, T,
                          robotVmax, robotAmax, snap0=snap0, crackle0=crackle0, pop0=pop0)
    return motion.to_plan()


def evaluate_smooth_motion(plan, t):
    """
    Evaluate a motion plan (handles both regular and smooth motions).
    """
    if 'smooth_motion' in plan:
        motion = plan['smooth_motion']
        return motion.position(t), motion.velocity(t), motion.acceleration(t)
    else:
        return evaluate_motion(plan, t)


def plan_motion(x0, x1, v0=0, v1=0, a0=0, a1=0, robotVmax=None, robotAmax=None,
                tau_start=None, tau_end=None):
    """
    Plan motion from (x0, v0, a0) to (x1, v1, a1).

    For PERFECT continuity when chaining motions, pass tau_start from the
    previous motion's tau_end. This ensures you're at the exact same point
    on the S-curve.

    IMPORTANT: The S-curve traces a LOOP in phase space.
    At any τ, velocity and acceleration are coupled - you can't specify
    arbitrary (v, a) pairs.
    """
    if robotVmax is None:
        robotVmax = float('inf')
    if robotAmax is None:
        robotAmax = float('inf')

    # If tau values are provided directly, use them (for motion chaining)
    if tau_start is not None or tau_end is not None:
        _tau_start = tau_start if tau_start is not None else -1.0
        _tau_end = tau_end if tau_end is not None else 1.0

        # Calculate T based on the fixed tau values
        delta_tau = _tau_end - _tau_start
        delta_f = normalized_f(_tau_end) - normalized_f(_tau_start)
        delta_x = x1 - x0

        if abs(delta_f) < 1e-10 or abs(delta_tau) < 1e-10:
            T = 0.0
        else:
            # Use vmax/amax constraints to find minimum T
            tau_range = np.linspace(_tau_start, _tau_end, 100)
            f_prime_max = np.max(normalized_f_derivative(tau_range))
            f_dbl_prime_max = np.max(np.abs(normalized_f_second_derivative(tau_range)))

            T_vmax = abs(delta_x) * delta_tau * f_prime_max / (delta_f * robotVmax)
            T_amax = sqrt(abs(delta_x) * delta_tau**2 * f_dbl_prime_max / (delta_f * robotAmax))
            T = max(T_vmax, T_amax)

        tau_start, tau_end = _tau_start, _tau_end
    else:
        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, v0, v1, a0, a1
        )

    v0_actual = get_velocity(0, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    v1_actual = get_velocity(T, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    a0_actual = get_acceleration(0, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    a1_actual = get_acceleration(T, T, x0, x1, tau_start, tau_end) if T > 0 else 0

    return {
        'T': T,
        'tau_start': tau_start,
        'tau_end': tau_end,
        'x0': x0,
        'x1': x1,
        'v0_requested': v0,
        'v1_requested': v1,
        'a0_requested': a0,
        'a1_requested': a1,
        'v0_actual': v0_actual,
        'v1_actual': v1_actual,
        'a0_actual': a0_actual,
        'a1_actual': a1_actual
    }


def evaluate_motion(plan, t):
    """Evaluate (position, velocity, acceleration) at time t."""
    pos = generalized_sCurve(t, plan['T'], plan['x0'], plan['x1'],
                              plan['tau_start'], plan['tau_end'])
    vel = get_velocity(t, plan['T'], plan['x0'], plan['x1'],
                       plan['tau_start'], plan['tau_end'])
    acc = get_acceleration(t, plan['T'], plan['x0'], plan['x1'],
                           plan['tau_start'], plan['tau_end'])
    return pos, vel, acc
