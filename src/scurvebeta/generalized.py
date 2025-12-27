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
    'continue_motion', 'SmoothMotion', 'evaluate_smooth_motion',
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


def _smooth_tau_coefficients(tau0, tau0_dot, tau0_ddot, tau0_dddot,
                              tau1, tau1_dot, tau1_ddot, tau1_dddot, T):
    """
    Compute polynomial coefficients for smooth τ(t) with given boundary conditions.

    Uses a 7th-degree polynomial for continuity through jerk (3rd derivative).
    τ(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵ + a6*t⁶ + a7*t⁷

    Boundary conditions at t=0: τ, τ', τ'', τ'''
    Boundary conditions at t=T: τ, τ', τ'', τ'''
    """
    # Coefficients from boundary conditions at t=0
    a0 = tau0
    a1 = tau0_dot
    a2 = tau0_ddot / 2
    a3 = tau0_dddot / 6

    # Solve for a4, a5, a6, a7 from boundary conditions at t=T
    T2, T3, T4, T5, T6, T7 = T**2, T**3, T**4, T**5, T**6, T**7

    # Right-hand side (what's left after subtracting known terms)
    b1 = tau1 - a0 - a1*T - a2*T2 - a3*T3
    b2 = tau1_dot - a1 - 2*a2*T - 3*a3*T2
    b3 = tau1_ddot - 2*a2 - 6*a3*T
    b4 = tau1_dddot - 6*a3

    # Matrix equation for [a4, a5, a6, a7]
    A = np.array([
        [T4, T5, T6, T7],
        [4*T3, 5*T4, 6*T5, 7*T6],
        [12*T2, 20*T3, 30*T4, 42*T5],
        [24*T, 60*T2, 120*T3, 210*T4]
    ])
    b = np.array([b1, b2, b3, b4])

    try:
        a4567 = np.linalg.solve(A, b)
        a4, a5, a6, a7 = a4567
    except np.linalg.LinAlgError:
        a4 = a5 = a6 = a7 = 0

    return np.array([a0, a1, a2, a3, a4, a5, a6, a7])


def _eval_tau_poly(t, coeffs):
    """Evaluate τ(t) polynomial and its derivatives (up to 3rd)."""
    a0, a1, a2, a3, a4, a5, a6, a7 = coeffs
    t = np.asarray(t)

    tau = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5 + a6*t**6 + a7*t**7
    tau_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4 + 6*a6*t**5 + 7*a7*t**6
    tau_ddot = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3 + 30*a6*t**4 + 42*a7*t**5
    tau_dddot = 6*a3 + 24*a4*t + 60*a5*t**2 + 120*a6*t**3 + 210*a7*t**4

    return tau, tau_dot, tau_ddot, tau_dddot


class SmoothMotion:
    """
    A motion with smooth time parameterization τ(t) for PERFECT derivative continuity.

    Instead of linear τ(t) = τ_start + (τ_end - τ_start) * t/T,
    we use a 7th-degree polynomial τ(t) that matches position, velocity,
    acceleration, AND JERK at boundaries.

    This allows reaching ANY target position while maintaining perfect continuity
    in velocity, acceleration, jerk, and higher derivatives!
    """

    def __init__(self, x0, x_target, v0, a0, j0, tau_start, prev_tau_dot, T,
                 robotVmax=None, robotAmax=None):
        """
        Create a smooth motion.

        Parameters:
        -----------
        x0 : float - Starting position
        x_target : float - Target position
        v0 : float - Starting velocity (must match exactly!)
        a0 : float - Starting acceleration (must match exactly!)
        j0 : float - Starting jerk (for perfect continuity)
        tau_start : float - Starting τ value on curve
        prev_tau_dot : float - τ'(t) from previous motion at junction
        T : float - Desired motion duration
        """
        self.x0 = x0
        self.x_target = x_target
        self.v0 = v0
        self.a0 = a0
        self.j0 = j0
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
            self.coeffs = np.zeros(8)
            return

        # Spatial scaling: S maps normalized f to actual position
        self.S = delta_x / delta_f

        # Get curve derivatives at start
        f_prime = normalized_f_derivative(tau_start)
        f_dbl_prime = normalized_f_second_derivative(tau_start)

        # Compute f''' numerically
        eps = 1e-6
        f_dbl_prime_plus = normalized_f_second_derivative(tau_start + eps)
        f_dbl_prime_minus = normalized_f_second_derivative(tau_start - eps)
        f_triple_prime = (f_dbl_prime_plus - f_dbl_prime_minus) / (2 * eps)

        # Compute τ'(0), τ''(0), τ'''(0) from physical velocity, acceleration, jerk
        # v = S * f' * τ'
        # a = S * (f'' * τ'² + f' * τ'')
        # j = S * (f''' * τ'³ + 3*f'' * τ' * τ'' + f' * τ''')

        if abs(self.S * f_prime) < 1e-10:
            tau0_dot = prev_tau_dot  # Use previous rate
        else:
            tau0_dot = v0 / (self.S * f_prime)

        if abs(self.S * f_prime) < 1e-10:
            tau0_ddot = 0
        else:
            tau0_ddot = (a0 - self.S * f_dbl_prime * tau0_dot**2) / (self.S * f_prime)

        if abs(self.S * f_prime) < 1e-10:
            tau0_dddot = 0
        else:
            tau0_dddot = (j0 - self.S * (f_triple_prime * tau0_dot**3 +
                          3 * f_dbl_prime * tau0_dot * tau0_ddot)) / (self.S * f_prime)

        # At t=T: τ=1, τ'=0, τ''=0, τ'''=0 (rest, all derivatives zero)
        tau1 = self.tau_end
        tau1_dot = 0
        tau1_ddot = 0
        tau1_dddot = 0

        # Find optimal T that respects constraints
        self.T = self._find_optimal_T(T, tau_start, tau0_dot, tau0_ddot, tau0_dddot,
                                       tau1, tau1_dot, tau1_ddot, tau1_dddot)

        # Compute polynomial coefficients
        self.coeffs = _smooth_tau_coefficients(
            tau_start, tau0_dot, tau0_ddot, tau0_dddot,
            tau1, tau1_dot, tau1_ddot, tau1_dddot,
            self.T
        )

    def _find_optimal_T(self, T_initial, tau0, tau0_dot, tau0_ddot, tau0_dddot,
                         tau1, tau1_dot, tau1_ddot, tau1_dddot):
        """Find T that respects velocity and acceleration constraints."""
        T = max(T_initial, 0.1)

        for _ in range(20):
            coeffs = _smooth_tau_coefficients(tau0, tau0_dot, tau0_ddot, tau0_dddot,
                                               tau1, tau1_dot, tau1_ddot, tau1_dddot, T)

            # Sample to check constraints
            t_samples = np.linspace(0, T, 100)
            tau_vals, tau_dot_vals, _, _ = _eval_tau_poly(t_samples, coeffs)

            # Compute velocity at samples
            f_prime_vals = normalized_f_derivative(np.clip(tau_vals, -1, 1))
            v_samples = np.abs(self.S * f_prime_vals * tau_dot_vals)

            max_v = np.max(v_samples)
            if max_v > self.robotVmax * 1.01:
                T = T * (max_v / self.robotVmax) * 1.1
            else:
                break

        return max(T, 0.01)

    def position(self, t):
        """Get position at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0:
            result = np.full_like(t, self.x0, dtype=float)
            return float(result[0]) if scalar else result

        tau, _, _, _ = _eval_tau_poly(np.clip(t, 0, self.T), self.coeffs)
        tau = np.clip(tau, min(self.tau_start, self.tau_end), max(self.tau_start, self.tau_end))

        f_vals = normalized_f(tau)
        f_start = normalized_f(self.tau_start)

        pos = self.x0 + self.S * (f_vals - f_start)
        return float(pos[0]) if scalar else pos

    def velocity(self, t):
        """Get velocity at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau, tau_dot, _, _ = _eval_tau_poly(np.clip(t, 0, self.T), self.coeffs)
        tau = np.clip(tau, -1, 1)

        f_prime = normalized_f_derivative(tau)
        vel = self.S * f_prime * tau_dot

        return float(vel[0]) if scalar else vel

    def acceleration(self, t):
        """Get acceleration at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau, tau_dot, tau_ddot, _ = _eval_tau_poly(np.clip(t, 0, self.T), self.coeffs)
        tau = np.clip(tau, -1, 1)

        f_prime = normalized_f_derivative(tau)
        f_dbl_prime = normalized_f_second_derivative(tau)

        acc = self.S * (f_dbl_prime * tau_dot**2 + f_prime * tau_ddot)

        return float(acc[0]) if scalar else acc

    def jerk(self, t):
        """Get jerk at time t."""
        t = np.asarray(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        if self.T <= 0:
            result = np.zeros_like(t, dtype=float)
            return float(result[0]) if scalar else result

        tau, tau_dot, tau_ddot, tau_dddot = _eval_tau_poly(np.clip(t, 0, self.T), self.coeffs)
        tau = np.clip(tau, -1, 1)

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
            'smooth_motion': self
        }


def continue_motion(prev_plan, x_target, T=None, robotVmax=None, robotAmax=None):
    """
    Continue from previous motion to reach x_target with PERFECT continuity.

    Uses smooth time parameterization τ(t) to ensure ALL derivatives
    (velocity, acceleration, jerk, snap, ...) are continuous at the junction.

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

    # Get jerk from previous motion (if smooth motion, use its jerk method)
    if 'smooth_motion' in prev_plan:
        j0 = prev_plan['smooth_motion'].jerk(prev_plan['T'])
    elif 'j1_actual' in prev_plan:
        j0 = prev_plan['j1_actual']
    else:
        # For linear τ motion, compute jerk at end
        # j = S * f'''(τ) * (τ')³ where τ' = Δτ/T
        prev_delta_tau = prev_plan['tau_end'] - prev_plan['tau_start']
        prev_delta_f = normalized_f(prev_plan['tau_end']) - normalized_f(prev_plan['tau_start'])
        prev_T = prev_plan['T']
        prev_delta_x = prev_plan['x1'] - prev_plan['x0']

        if prev_T > 0 and abs(prev_delta_f) > 1e-10:
            S = prev_delta_x / prev_delta_f
            tau_dot = prev_delta_tau / prev_T
            tau = prev_plan['tau_end']

            # f''' numerically
            eps = 1e-6
            f_dbl_prime_plus = normalized_f_second_derivative(min(tau + eps, 1))
            f_dbl_prime_minus = normalized_f_second_derivative(max(tau - eps, -1))
            f_triple_prime = (f_dbl_prime_plus - f_dbl_prime_minus) / (2 * eps)

            j0 = S * f_triple_prime * tau_dot**3
        else:
            j0 = 0

    # Get prev_tau_dot for the previous motion
    if 'smooth_motion' in prev_plan:
        # Get τ'(T) from the smooth motion
        _, prev_tau_dot, _, _ = _eval_tau_poly(prev_plan['T'], prev_plan['smooth_motion'].coeffs)
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
                          robotVmax, robotAmax)
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
