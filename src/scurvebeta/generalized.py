"""
Generalized S-curve motion with arbitrary boundary conditions.

This module extends the basic S-curve to support arbitrary initial and final
states (position, velocity, acceleration). Instead of always starting and
ending at rest, you can now specify any point on the S-curve as your
start/end state.

The key insight: the S-curve already contains ALL possible motion states
along its path. We just need to find where on the curve our desired
boundary conditions occur and use that segment.

Mathematical foundation:
- The normalized S-curve f(τ) goes from τ=-1 to τ=1
- At τ=-1: position=0, velocity=0, acceleration=0 (rest, start)
- At τ=0:  position=0.5, velocity=MAX, acceleration=0 (peak velocity)
- At τ=+1: position=1, velocity=0, acceleration=0 (rest, end)

When using a segment [τ_start, τ_end] mapped to [0, T] and [x0, x1]:
- position(t) = x0 + (x1-x0) * (f(τ(t)) - f(τ_start)) / (f(τ_end) - f(τ_start))
- velocity(t) = (x1-x0) * Δτ / (Δf * T) * f'(τ(t))
- acceleration(t) = (x1-x0) * Δτ² / (Δf * T²) * f''(τ(t))
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
    'MAX_NORMALIZED_VEL', 'MAX_NORMALIZED_ACC'
]

# Constants for the beta S-curve with p=2.5
MAX_NORMALIZED_VEL = 16 / (5 * pi)  # ≈ 1.0186 (max of f'(τ) at τ=0)
MAX_NORMALIZED_ACC = 3 * sqrt(3) / pi  # ≈ 1.6540 (max of f''(τ) at τ=-0.5)

# Precomputed coefficient for beta function
BETA_COEF = 0.5092958178940651  # 1 / (2 * beta(0.5, 3.5))


def normalized_f(tau):
    """
    Normalized S-curve position function f(τ) for τ ∈ [-1, 1].
    Returns position in range [0, 1].
    """
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

        neg_mask = t_masked < 0
        pos_mask = t_masked > 0
        zero_mask = t_masked == 0

        result_masked = np.zeros_like(t_masked)
        if np.any(neg_mask):
            result_masked[neg_mask] = 0.5 - BETA_COEF * betainc(0.5, 3.5, t_masked[neg_mask]**2) * beta_val
        if np.any(pos_mask):
            result_masked[pos_mask] = 0.5 + BETA_COEF * betainc(0.5, 3.5, t_masked[pos_mask]**2) * beta_val
        if np.any(zero_mask):
            result_masked[zero_mask] = 0.5
        result[mask] = result_masked

    return float(result[0]) if scalar_input else result


def normalized_f_derivative(tau):
    """
    First derivative: f'(τ) = velocity profile.
    f'(τ) = (1 - τ²)^2.5 / Beta(0.5, 3.5) for |τ| < 1
    Max ≈ 1.0186 at τ=0.
    """
    tau = np.asarray(tau, dtype=float)
    scalar_input = tau.ndim == 0
    tau = np.atleast_1d(tau)

    result = np.zeros_like(tau)
    mask = np.abs(tau) < 1

    if np.any(mask):
        t_masked = tau[mask]
        beta_val = beta_func(0.5, 3.5)
        result[mask] = np.power(1 - t_masked**2, 2.5) / beta_val

    return float(result[0]) if scalar_input else result


def normalized_f_second_derivative(tau):
    """
    Second derivative: f''(τ) = acceleration profile.
    f''(τ) = -5τ(1 - τ²)^1.5 / Beta(0.5, 3.5)
    Max ≈ 1.654 at τ=-0.5, min ≈ -1.654 at τ=0.5.
    """
    tau = np.asarray(tau, dtype=float)
    scalar_input = tau.ndim == 0
    tau = np.atleast_1d(tau)

    result = np.zeros_like(tau)
    mask = np.abs(tau) < 1

    if np.any(mask):
        t_masked = tau[mask]
        beta_val = beta_func(0.5, 3.5)
        result[mask] = -5 * t_masked * np.power(1 - t_masked**2, 1.5) / beta_val

    return float(result[0]) if scalar_input else result


def find_tau_for_velocity(v_normalized, accelerating=True):
    """
    Find τ value where normalized velocity equals v_normalized.

    Parameters:
    -----------
    v_normalized : float
        Normalized velocity (0 to 1, where 1 = max normalized velocity)
    accelerating : bool
        If True, find τ in [-1, 0] (accelerating phase, positive acceleration)
        If False, find τ in [0, 1] (decelerating phase, negative acceleration)

    Returns:
    --------
    tau : float
        The τ value where f'(τ)/MAX_NORMALIZED_VEL ≈ v_normalized
    """
    if v_normalized <= 0:
        return -1.0 if accelerating else 1.0
    if v_normalized >= 1:
        return 0.0

    target_vel = v_normalized * MAX_NORMALIZED_VEL

    def objective(tau):
        return normalized_f_derivative(tau) - target_vel

    if accelerating:
        return brentq(objective, -1 + 1e-10, 0)
    else:
        return brentq(objective, 0, 1 - 1e-10)


def _compute_scaling_factor(tau_start, tau_end):
    """
    Compute the position scaling factor Δf = f(τ_end) - f(τ_start).
    """
    return normalized_f(tau_end) - normalized_f(tau_start)


def generalized_motion_time(robotVmax, robotAmax, x0, x1, v0=0, v1=0, a0=0, a1=0):
    """
    Calculate motion time for generalized boundary conditions.

    This finds the segment [τ_start, τ_end] of the S-curve and the time T
    such that the motion matches the specified boundary conditions while
    respecting velocity and acceleration limits.

    Parameters:
    -----------
    robotVmax : float
        Maximum allowed velocity (magnitude)
    robotAmax : float
        Maximum allowed acceleration (magnitude)
    x0, x1 : float
        Start and end positions
    v0, v1 : float
        Start and end velocities (default 0 for rest)
        Sign indicates direction: positive = towards increasing position
    a0, a1 : float
        Start and end accelerations (default 0)
        Positive = increasing velocity, negative = decreasing velocity

    Returns:
    --------
    T : float
        Motion time in seconds
    tau_start : float
        Starting point on normalized curve [-1, 1]
    tau_end : float
        Ending point on normalized curve [-1, 1]
    """
    delta_x = x1 - x0
    motion_direction = 1 if delta_x >= 0 else -1

    # Handle zero motion case
    if abs(delta_x) < 1e-10:
        return 0.0, -1.0, 1.0

    # ===== STEP 1: Determine τ_start and τ_end from boundary velocities =====

    # The velocity on the actual curve is:
    # v(t) = motion_direction * (|Δx| * Δτ / (Δf * T)) * f'(τ)
    #
    # At boundaries, we need:
    # v0 = motion_direction * K * f'(τ_start)
    # v1 = motion_direction * K * f'(τ_end)
    # where K = |Δx| * Δτ / (Δf * T) > 0
    #
    # The sign of v0 relative to motion_direction tells us which phase we're in

    # Normalize velocities (make them relative to motion direction)
    v0_rel = v0 * motion_direction  # Positive if velocity is in motion direction
    v1_rel = v1 * motion_direction

    # For the S-curve, velocity is always positive (f'(τ) >= 0)
    # So v_rel must be >= 0 for a valid motion on this segment
    # If v_rel < 0, the object is moving opposite to the intended direction

    # Determine τ_start
    if abs(v0) < 1e-10 and abs(a0) < 1e-10:
        # Rest at start
        tau_start = -1.0
    elif v0_rel >= 0:
        # Moving in the same direction as the motion (or at rest)
        # Find τ where velocity matches
        # Need to determine if accelerating or decelerating based on a0
        if a0 >= 0:
            # Accelerating phase: τ in [-1, 0]
            accelerating = True
        else:
            # Decelerating phase: τ in [0, 1]
            accelerating = False

        # We need to find τ such that the velocity ratio matches
        # But we don't know K yet, so we'll iterate
        # For now, use a heuristic based on relative velocity
        tau_start = -1.0  # Default, will be refined
    else:
        # Moving opposite to motion direction - not supported with single segment
        # Default to rest
        tau_start = -1.0

    # Determine τ_end
    if abs(v1) < 1e-10 and abs(a1) < 1e-10:
        # Rest at end
        tau_end = 1.0
    elif v1_rel >= 0:
        if a1 <= 0:
            # Decelerating at end: τ in [0, 1]
            accelerating = False
        else:
            # Accelerating at end: τ in [-1, 0]
            accelerating = True
        tau_end = 1.0  # Default, will be refined
    else:
        tau_end = 1.0

    # ===== STEP 2: Iterative refinement to match velocities =====

    # For rest-to-rest, use standard calculation
    if abs(v0) < 1e-10 and abs(v1) < 1e-10:
        tau_start = -1.0
        tau_end = 1.0
        motionRange = abs(delta_x)
        T = max(
            2.572148274314975138567 * sqrt(motionRange / robotAmax),
            2.037183271576260297842 * motionRange / robotVmax
        )
        return T, tau_start, tau_end

    # For non-rest boundaries, we need to solve for τ_start, τ_end, T simultaneously
    #
    # Constraints:
    # 1. v0 = motion_dir * (|Δx| * Δτ / (Δf * T)) * f'(τ_start)
    # 2. v1 = motion_dir * (|Δx| * Δτ / (Δf * T)) * f'(τ_end)
    # 3. max|v(t)| <= robotVmax
    # 4. max|a(t)| <= robotAmax

    # From (1) and (2):
    # v0 / v1 = f'(τ_start) / f'(τ_end)  (if both non-zero)

    # Case: v0 != 0, v1 = 0 (moving start, rest at end)
    if abs(v0) > 1e-10 and abs(v1) < 1e-10:
        tau_end = 1.0  # Rest at end
        f_prime_end = 0.0

        # v0 determines τ_start
        # We need: v0_rel = K * f'(τ_start) where K = |Δx| * Δτ / (Δf * T)
        # The phase (accelerating or decelerating) is determined by a0

        # If |v0| <= robotVmax, we can find a valid configuration
        # τ_start is in accelerating phase if a0 > 0 (or we're at start of accel)
        accelerating_start = (a0 > 0) or (a0 == 0 and v0_rel < robotVmax * 0.99)

        # We'll solve iteratively
        # For a given τ_start, T is determined by the velocity constraint at start
        # T = |Δx| * Δτ * f'(τ_start) / (Δf * v0_rel)

        best_tau_start = -1.0
        best_T = float('inf')

        # Search for valid τ_start
        for tau_s in np.linspace(-0.999, 0.999, 200):
            if tau_s >= tau_end:
                continue

            delta_tau = tau_end - tau_s
            delta_f = _compute_scaling_factor(tau_s, tau_end)

            if delta_f <= 0:
                continue

            f_prime_start = normalized_f_derivative(tau_s)

            if f_prime_start < 1e-10:
                continue

            # T from velocity at start
            T_from_v0 = abs(delta_x) * delta_tau * f_prime_start / (delta_f * abs(v0_rel))

            # Check velocity constraint (max velocity on segment)
            tau_range = np.linspace(tau_s, tau_end, 50)
            f_prime_max = np.max(normalized_f_derivative(tau_range))
            max_vel = abs(delta_x) * delta_tau * f_prime_max / (delta_f * T_from_v0)

            if max_vel > robotVmax * 1.01:
                continue

            # Check acceleration constraint
            f_double_prime_max = np.max(np.abs(normalized_f_second_derivative(tau_range)))
            max_acc = abs(delta_x) * delta_tau**2 * f_double_prime_max / (delta_f * T_from_v0**2)

            if max_acc > robotAmax * 1.01:
                continue

            # Check that start velocity sign matches expectation
            actual_v0 = motion_direction * abs(delta_x) * delta_tau * f_prime_start / (delta_f * T_from_v0)
            if abs(actual_v0 - v0) > abs(v0) * 0.1:
                continue

            if T_from_v0 < best_T:
                best_T = T_from_v0
                best_tau_start = tau_s

        return best_T, best_tau_start, tau_end

    # Case: v0 = 0, v1 != 0 (rest at start, moving at end)
    if abs(v0) < 1e-10 and abs(v1) > 1e-10:
        tau_start = -1.0  # Rest at start

        accelerating_end = (a1 > 0) or (a1 == 0)

        best_tau_end = 1.0
        best_T = float('inf')

        for tau_e in np.linspace(-0.999, 0.999, 200):
            if tau_e <= tau_start:
                continue

            delta_tau = tau_e - tau_start
            delta_f = _compute_scaling_factor(tau_start, tau_e)

            if delta_f <= 0:
                continue

            f_prime_end = normalized_f_derivative(tau_e)

            if f_prime_end < 1e-10:
                continue

            T_from_v1 = abs(delta_x) * delta_tau * f_prime_end / (delta_f * abs(v1_rel))

            tau_range = np.linspace(tau_start, tau_e, 50)
            f_prime_max = np.max(normalized_f_derivative(tau_range))
            max_vel = abs(delta_x) * delta_tau * f_prime_max / (delta_f * T_from_v1)

            if max_vel > robotVmax * 1.01:
                continue

            f_double_prime_max = np.max(np.abs(normalized_f_second_derivative(tau_range)))
            max_acc = abs(delta_x) * delta_tau**2 * f_double_prime_max / (delta_f * T_from_v1**2)

            if max_acc > robotAmax * 1.01:
                continue

            actual_v1 = motion_direction * abs(delta_x) * delta_tau * f_prime_end / (delta_f * T_from_v1)
            if abs(actual_v1 - v1) > abs(v1) * 0.1:
                continue

            if T_from_v1 < best_T:
                best_T = T_from_v1
                best_tau_end = tau_e

        return best_T, tau_start, best_tau_end

    # Case: both v0 != 0 and v1 != 0
    # Need to find τ_start, τ_end such that f'(τ_start)/f'(τ_end) = v0_rel/v1_rel
    velocity_ratio = abs(v0_rel / v1_rel) if abs(v1_rel) > 1e-10 else float('inf')

    best_tau_start = -1.0
    best_tau_end = 1.0
    best_T = float('inf')

    for tau_s in np.linspace(-0.999, 0.5, 100):
        f_prime_s = normalized_f_derivative(tau_s)
        if f_prime_s < 1e-10:
            continue

        target_f_prime_e = f_prime_s / velocity_ratio

        if target_f_prime_e > MAX_NORMALIZED_VEL or target_f_prime_e < 1e-10:
            continue

        # Find τ_end with this f' value
        for tau_e in np.linspace(tau_s + 0.01, 0.999, 100):
            f_prime_e = normalized_f_derivative(tau_e)

            if abs(f_prime_e - target_f_prime_e) > 0.05 * MAX_NORMALIZED_VEL:
                continue

            delta_tau = tau_e - tau_s
            delta_f = _compute_scaling_factor(tau_s, tau_e)

            if delta_f <= 0:
                continue

            T_from_v0 = abs(delta_x) * delta_tau * f_prime_s / (delta_f * abs(v0_rel))

            tau_range = np.linspace(tau_s, tau_e, 50)
            f_prime_max = np.max(normalized_f_derivative(tau_range))
            max_vel = abs(delta_x) * delta_tau * f_prime_max / (delta_f * T_from_v0)

            if max_vel > robotVmax * 1.01:
                continue

            f_double_prime_max = np.max(np.abs(normalized_f_second_derivative(tau_range)))
            max_acc = abs(delta_x) * delta_tau**2 * f_double_prime_max / (delta_f * T_from_v0**2)

            if max_acc > robotAmax * 1.01:
                continue

            if T_from_v0 < best_T:
                best_T = T_from_v0
                best_tau_start = tau_s
                best_tau_end = tau_e

    if best_T == float('inf'):
        # Fallback to rest-to-rest
        tau_start = -1.0
        tau_end = 1.0
        motionRange = abs(delta_x)
        T = max(
            2.572148274314975138567 * sqrt(motionRange / robotAmax),
            2.037183271576260297842 * motionRange / robotVmax
        )
        return T, tau_start, tau_end

    return best_T, best_tau_start, best_tau_end


def generalized_sCurve(t, T, x0, x1, tau_start=-1, tau_end=1):
    """
    Generalized S-curve position at time t.
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return np.full_like(t, x0) if not scalar_input else x0

    tau = tau_start + (tau_end - tau_start) * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_values = normalized_f(tau)
    f_start = normalized_f(tau_start)
    f_end = normalized_f(tau_end)
    delta_f = f_end - f_start

    if abs(delta_f) < 1e-10:
        position = np.full_like(t, x0)
    else:
        position = x0 + (x1 - x0) * (f_values - f_start) / delta_f

    return float(position[0]) if scalar_input else position


def get_velocity(t, T, x0, x1, tau_start=-1, tau_end=1):
    """
    Get velocity at time t for generalized S-curve motion.
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return np.zeros_like(t) if not scalar_input else 0.0

    delta_tau = tau_end - tau_start
    delta_x = x1 - x0
    delta_f = _compute_scaling_factor(tau_start, tau_end)

    if abs(delta_f) < 1e-10:
        return np.zeros_like(t) if not scalar_input else 0.0

    tau = tau_start + delta_tau * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_prime = normalized_f_derivative(tau)
    velocity = (delta_x / delta_f) * f_prime * (delta_tau / T)

    return float(velocity[0]) if scalar_input else velocity


def get_acceleration(t, T, x0, x1, tau_start=-1, tau_end=1):
    """
    Get acceleration at time t for generalized S-curve motion.
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    if T <= 0:
        return np.zeros_like(t) if not scalar_input else 0.0

    delta_tau = tau_end - tau_start
    delta_x = x1 - x0
    delta_f = _compute_scaling_factor(tau_start, tau_end)

    if abs(delta_f) < 1e-10:
        return np.zeros_like(t) if not scalar_input else 0.0

    tau = tau_start + delta_tau * t / T
    tau = np.clip(tau, min(tau_start, tau_end), max(tau_start, tau_end))

    f_double_prime = normalized_f_second_derivative(tau)
    acceleration = (delta_x / delta_f) * f_double_prime * (delta_tau / T)**2

    return float(acceleration[0]) if scalar_input else acceleration


def plan_motion(x0, x1, v0=0, v1=0, a0=0, a1=0, robotVmax=None, robotAmax=None):
    """
    Plan a motion from state (x0, v0, a0) to state (x1, v1, a1).

    Returns a dict with motion parameters that can be used with evaluate_motion().
    """
    if robotVmax is None:
        robotVmax = float('inf')
    if robotAmax is None:
        robotAmax = float('inf')

    T, tau_start, tau_end = generalized_motion_time(
        robotVmax, robotAmax, x0, x1, v0, v1, a0, a1
    )

    # Calculate actual boundary values
    actual_v0 = get_velocity(0, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    actual_v1 = get_velocity(T, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    actual_a0 = get_acceleration(0, T, x0, x1, tau_start, tau_end) if T > 0 else 0
    actual_a1 = get_acceleration(T, T, x0, x1, tau_start, tau_end) if T > 0 else 0

    return {
        'T': T,
        'tau_start': tau_start,
        'tau_end': tau_end,
        'x0': x0,
        'x1': x1,
        'v0_requested': v0,
        'v1_requested': v1,
        'v0_actual': actual_v0,
        'v1_actual': actual_v1,
        'a0_actual': actual_a0,
        'a1_actual': actual_a1
    }


def evaluate_motion(plan, t):
    """
    Evaluate position, velocity, and acceleration at time t given a motion plan.
    """
    pos = generalized_sCurve(
        t, plan['T'], plan['x0'], plan['x1'],
        plan['tau_start'], plan['tau_end']
    )
    vel = get_velocity(
        t, plan['T'], plan['x0'], plan['x1'],
        plan['tau_start'], plan['tau_end']
    )
    acc = get_acceleration(
        t, plan['T'], plan['x0'], plan['x1'],
        plan['tau_start'], plan['tau_end']
    )
    return pos, vel, acc
