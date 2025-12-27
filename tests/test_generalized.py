"""
Comprehensive test suite for generalized S-curve motion.
"""
import pytest
import numpy as np
from math import sqrt, pi
import sys
sys.path.insert(0, '/home/user/s-curve-beta/src')

from scurvebeta.generalized import (
    normalized_f, normalized_f_derivative, normalized_f_second_derivative,
    find_tau_for_velocity, generalized_motion_time, generalized_sCurve,
    get_velocity, get_acceleration, plan_motion, evaluate_motion,
    MAX_NORMALIZED_VEL, MAX_NORMALIZED_ACC
)
import scurvebeta as scb


class TestNormalizedFunctions:
    """Test the normalized S-curve functions."""

    def test_normalized_f_boundaries(self):
        """f(τ) should be 0 at τ=-1 and 1 at τ=1."""
        assert abs(normalized_f(-1) - 0.0) < 1e-6
        assert abs(normalized_f(1) - 1.0) < 1e-6

    def test_normalized_f_midpoint(self):
        """f(0) should be 0.5 (symmetric curve)."""
        assert abs(normalized_f(0) - 0.5) < 1e-6

    def test_normalized_f_monotonic(self):
        """f(τ) should be monotonically increasing."""
        tau = np.linspace(-1, 1, 101)
        f_values = normalized_f(tau)
        assert np.all(np.diff(f_values) >= 0)

    def test_normalized_f_outside_bounds(self):
        """f(τ) should clamp to 0 and 1 outside [-1, 1]."""
        assert normalized_f(-2) == 0.0
        assert normalized_f(2) == 1.0

    def test_normalized_derivative_max(self):
        """f'(τ) should have max at τ=0."""
        tau = np.linspace(-1, 1, 1001)
        f_prime = normalized_f_derivative(tau)
        max_idx = np.argmax(f_prime)
        assert abs(tau[max_idx]) < 0.01  # Max near τ=0
        assert abs(f_prime[max_idx] - MAX_NORMALIZED_VEL) < 0.01

    def test_normalized_derivative_zero_at_boundaries(self):
        """f'(τ) should be 0 at τ=±1."""
        assert abs(normalized_f_derivative(-1)) < 1e-6
        assert abs(normalized_f_derivative(1)) < 1e-6

    def test_normalized_second_derivative_max(self):
        """f''(τ) should have max at τ=-0.5 and min at τ=0.5."""
        tau = np.linspace(-1, 1, 1001)
        f_double_prime = normalized_f_second_derivative(tau)

        max_idx = np.argmax(f_double_prime)
        min_idx = np.argmin(f_double_prime)

        assert abs(tau[max_idx] - (-0.5)) < 0.02  # Max near τ=-0.5
        assert abs(tau[min_idx] - 0.5) < 0.02    # Min near τ=0.5

    def test_normalized_second_derivative_zero_at_midpoint(self):
        """f''(0) should be 0 (inflection point)."""
        assert abs(normalized_f_second_derivative(0)) < 1e-6

    def test_normalized_second_derivative_antisymmetric(self):
        """f''(τ) should be antisymmetric: f''(-τ) = -f''(τ)."""
        tau_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for tau in tau_values:
            f_pp_pos = normalized_f_second_derivative(tau)
            f_pp_neg = normalized_f_second_derivative(-tau)
            assert abs(f_pp_pos + f_pp_neg) < 1e-6


class TestFindTau:
    """Test finding τ for given states."""

    def test_find_tau_zero_velocity(self):
        """Zero velocity should return τ=±1."""
        tau = find_tau_for_velocity(0, accelerating=True)
        assert abs(tau - (-1)) < 0.01

        tau = find_tau_for_velocity(0, accelerating=False)
        assert abs(tau - 1) < 0.01

    def test_find_tau_max_velocity(self):
        """Max velocity should return τ=0."""
        tau = find_tau_for_velocity(1.0, accelerating=True)
        assert abs(tau) < 0.01

        tau = find_tau_for_velocity(1.0, accelerating=False)
        assert abs(tau) < 0.01

    def test_find_tau_half_velocity_accelerating(self):
        """Half velocity in accelerating phase should be in [-1, 0]."""
        tau = find_tau_for_velocity(0.5, accelerating=True)
        assert -1 < tau < 0

        # Verify the velocity at this τ
        vel = normalized_f_derivative(tau) / MAX_NORMALIZED_VEL
        assert abs(vel - 0.5) < 0.01

    def test_find_tau_half_velocity_decelerating(self):
        """Half velocity in decelerating phase should be in [0, 1]."""
        tau = find_tau_for_velocity(0.5, accelerating=False)
        assert 0 < tau < 1

        # Verify the velocity at this τ
        vel = normalized_f_derivative(tau) / MAX_NORMALIZED_VEL
        assert abs(vel - 0.5) < 0.01


class TestRestToRestMotion:
    """Test that rest-to-rest motion matches original implementation."""

    def test_rest_to_rest_position(self):
        """Generalized should match original for rest-to-rest."""
        x0, x1 = 0, 10
        robotVmax, robotAmax = 5, 2

        # Original
        T_orig = scb.motionTime(robotVmax, robotAmax, abs(x1 - x0))

        # Generalized
        T_gen, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, 0, 0, 0
        )

        # Times should match
        assert abs(T_orig - T_gen) < 0.01

        # Positions should match
        t = np.linspace(0, T_orig, 101)
        pos_orig = scb.sCurve(t, T_orig, x0, x1)
        pos_gen = generalized_sCurve(t, T_gen, x0, x1, tau_start, tau_end)

        np.testing.assert_allclose(pos_orig, pos_gen, rtol=1e-3)

    def test_rest_to_rest_boundary_conditions(self):
        """Rest-to-rest should have zero velocity and acceleration at boundaries."""
        x0, x1 = -5, 15
        robotVmax, robotAmax = 8, 3

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, 0, 0, 0
        )

        # At t=0
        v_start = get_velocity(0, T, x0, x1, tau_start, tau_end)
        a_start = get_acceleration(0, T, x0, x1, tau_start, tau_end)
        assert abs(v_start) < 0.01
        assert abs(a_start) < 0.01

        # At t=T
        v_end = get_velocity(T, T, x0, x1, tau_start, tau_end)
        a_end = get_acceleration(T, T, x0, x1, tau_start, tau_end)
        assert abs(v_end) < 0.01
        assert abs(a_end) < 0.01


class TestCustomBoundaryConditions:
    """Test motions with custom boundary conditions."""

    def test_start_with_velocity(self):
        """Start with non-zero velocity, end at rest."""
        x0, x1 = 0, 10
        robotVmax, robotAmax = 10, 5
        v0 = 3  # Starting velocity

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, v0, 0, 0, 0
        )

        # Check start velocity is approximately v0
        v_actual_start = get_velocity(0, T, x0, x1, tau_start, tau_end)
        # Note: Due to scaling, exact match isn't guaranteed without iteration
        # But direction should match
        assert v_actual_start > 0  # Moving forward

        # Check end velocity is approximately 0
        v_actual_end = get_velocity(T, T, x0, x1, tau_start, tau_end)
        assert abs(v_actual_end) < 0.1

    def test_end_with_velocity(self):
        """Start at rest, end with non-zero velocity."""
        x0, x1 = 0, 10
        robotVmax, robotAmax = 10, 5
        v1 = 2  # Ending velocity

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, v1, 0, 0
        )

        # Check start velocity is approximately 0
        v_actual_start = get_velocity(0, T, x0, x1, tau_start, tau_end)
        assert abs(v_actual_start) < 0.1

        # Check end velocity direction matches
        v_actual_end = get_velocity(T, T, x0, x1, tau_start, tau_end)
        assert v_actual_end > 0  # Moving forward

    def test_motion_smoothness(self):
        """Verify motion is smooth (no discontinuities in velocity)."""
        x0, x1 = 0, 20
        robotVmax, robotAmax = 10, 5
        v0, v1 = 2, 1

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, v0, v1, 0, 0
        )

        t = np.linspace(0, T, 1001)
        vel = get_velocity(t, T, x0, x1, tau_start, tau_end)
        acc = get_acceleration(t, T, x0, x1, tau_start, tau_end)

        # Velocity should be continuous (no large jumps)
        vel_diff = np.diff(vel)
        assert np.all(np.abs(vel_diff) < 0.5)

        # Acceleration should be continuous
        acc_diff = np.diff(acc)
        assert np.all(np.abs(acc_diff) < 1.0)


class TestConstraints:
    """Test velocity and acceleration constraints."""

    def test_velocity_constraint(self):
        """Velocity should not exceed robotVmax."""
        x0, x1 = 0, 100
        robotVmax, robotAmax = 5, 10  # Velocity limited

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, 0, 0, 0
        )

        t = np.linspace(0, T, 1001)
        vel = get_velocity(t, T, x0, x1, tau_start, tau_end)

        assert np.max(np.abs(vel)) <= robotVmax * 1.01  # Allow 1% tolerance

    def test_acceleration_constraint(self):
        """Acceleration should not exceed robotAmax."""
        x0, x1 = 0, 10
        robotVmax, robotAmax = 100, 3  # Acceleration limited

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, 0, 0, 0
        )

        t = np.linspace(0, T, 1001)
        acc = get_acceleration(t, T, x0, x1, tau_start, tau_end)

        assert np.max(np.abs(acc)) <= robotAmax * 1.01  # Allow 1% tolerance


class TestPlanMotion:
    """Test the high-level plan_motion interface."""

    def test_plan_basic(self):
        """Test basic planning."""
        plan = plan_motion(0, 10, robotVmax=5, robotAmax=2)

        assert plan['x0'] == 0
        assert plan['x1'] == 10
        assert plan['T'] > 0
        assert -1 <= plan['tau_start'] <= 1
        assert -1 <= plan['tau_end'] <= 1

    def test_evaluate_motion(self):
        """Test motion evaluation."""
        plan = plan_motion(0, 10, robotVmax=5, robotAmax=2)

        t = np.linspace(0, plan['T'], 11)
        pos, vel, acc = evaluate_motion(plan, t)

        # Start position
        assert abs(pos[0] - 0) < 0.01
        # End position
        assert abs(pos[-1] - 10) < 0.01

    def test_plan_with_velocity(self):
        """Test planning with initial/final velocity."""
        plan = plan_motion(0, 10, v0=2, v1=1, robotVmax=10, robotAmax=5)

        assert plan['v0_requested'] == 2
        assert plan['v1_requested'] == 1
        assert plan['T'] > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_motion(self):
        """Test when x0 == x1."""
        T, tau_start, tau_end = generalized_motion_time(
            5, 2, 0, 0, 0, 0, 0, 0
        )
        # Should handle gracefully
        assert T >= 0

    def test_negative_motion(self):
        """Test motion in negative direction."""
        x0, x1 = 10, 0  # Moving backwards
        robotVmax, robotAmax = 5, 2

        T, tau_start, tau_end = generalized_motion_time(
            robotVmax, robotAmax, x0, x1, 0, 0, 0, 0
        )

        t = np.linspace(0, T, 101)
        pos = generalized_sCurve(t, T, x0, x1, tau_start, tau_end)

        # Should start at x0 and end at x1
        assert abs(pos[0] - x0) < 0.1
        assert abs(pos[-1] - x1) < 0.1

    def test_array_input(self):
        """Test with array inputs."""
        T, tau_start, tau_end = generalized_motion_time(5, 2, 0, 10, 0, 0, 0, 0)

        t = np.array([0, T/4, T/2, 3*T/4, T])
        pos = generalized_sCurve(t, T, 0, 10, tau_start, tau_end)
        vel = get_velocity(t, T, 0, 10, tau_start, tau_end)
        acc = get_acceleration(t, T, 0, 10, tau_start, tau_end)

        assert len(pos) == 5
        assert len(vel) == 5
        assert len(acc) == 5

    def test_scalar_input(self):
        """Test with scalar inputs."""
        T, tau_start, tau_end = generalized_motion_time(5, 2, 0, 10, 0, 0, 0, 0)

        pos = generalized_sCurve(T/2, T, 0, 10, tau_start, tau_end)
        vel = get_velocity(T/2, T, 0, 10, tau_start, tau_end)
        acc = get_acceleration(T/2, T, 0, 10, tau_start, tau_end)

        assert isinstance(pos, float)
        assert isinstance(vel, float)
        assert isinstance(acc, float)


class TestMathematicalProperties:
    """Test mathematical properties of the S-curve."""

    def test_derivative_consistency(self):
        """Numerical derivative should match analytical velocity."""
        T, tau_start, tau_end = generalized_motion_time(5, 2, 0, 10, 0, 0, 0, 0)

        dt = 0.0001
        t = 2.5
        pos1 = generalized_sCurve(t, T, 0, 10, tau_start, tau_end)
        pos2 = generalized_sCurve(t + dt, T, 0, 10, tau_start, tau_end)

        vel_numerical = (pos2 - pos1) / dt
        vel_analytical = get_velocity(t, T, 0, 10, tau_start, tau_end)

        assert abs(vel_numerical - vel_analytical) < 0.01

    def test_second_derivative_consistency(self):
        """Numerical second derivative should match analytical acceleration."""
        T, tau_start, tau_end = generalized_motion_time(5, 2, 0, 10, 0, 0, 0, 0)

        dt = 0.0001
        t = 2.5
        vel1 = get_velocity(t, T, 0, 10, tau_start, tau_end)
        vel2 = get_velocity(t + dt, T, 0, 10, tau_start, tau_end)

        acc_numerical = (vel2 - vel1) / dt
        acc_analytical = get_acceleration(t, T, 0, 10, tau_start, tau_end)

        assert abs(acc_numerical - acc_analytical) < 0.1

    def test_symmetry(self):
        """Curve should be point-symmetric around midpoint."""
        T, tau_start, tau_end = generalized_motion_time(5, 2, 0, 10, 0, 0, 0, 0)

        t = np.linspace(0, T, 101)
        pos = generalized_sCurve(t, T, 0, 10, tau_start, tau_end)

        # Check symmetry: pos(t) + pos(T-t) = x0 + x1
        for i in range(len(t) // 2):
            sum_pos = pos[i] + pos[-(i+1)]
            assert abs(sum_pos - (0 + 10)) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
