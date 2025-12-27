"""
Generate demonstration plots for generalized S-curve motion.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/user/s-curve-beta/src')

from scurvebeta.generalized import (
    normalized_f, normalized_f_derivative, normalized_f_second_derivative,
    generalized_motion_time, generalized_sCurve, get_velocity, get_acceleration,
    plan_motion, evaluate_motion, MAX_NORMALIZED_VEL, MAX_NORMALIZED_ACC
)
import scurvebeta as scb

# Create img directory if needed
import os
os.makedirs('/home/user/s-curve-beta/img', exist_ok=True)


def plot_normalized_curve():
    """Plot the normalized S-curve and its derivatives."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    tau = np.linspace(-1, 1, 1001)
    pos = normalized_f(tau)
    vel = normalized_f_derivative(tau)
    acc = normalized_f_second_derivative(tau)

    # Jerk via numerical differentiation
    dtau = tau[1] - tau[0]
    jerk = np.gradient(acc, dtau)

    axs[0].plot(tau, pos, 'b-', linewidth=2)
    axs[0].set_ylabel('Position f(τ)', fontsize=11)
    axs[0].set_title('Normalized S-Curve (Beta Function, p=2.5)', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axs[0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axs[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axs[0].set_xlim(-1, 1)

    axs[1].plot(tau, vel, 'g-', linewidth=2)
    axs[1].set_ylabel("Velocity f'(τ)", fontsize=11)
    axs[1].grid(True, alpha=0.3)
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axs[1].axhline(y=MAX_NORMALIZED_VEL, color='r', linestyle='--', alpha=0.5)
    axs[1].axvline(x=0, color='r', linestyle='--', alpha=0.3)
    axs[1].annotate('max velocity', xy=(0.05, MAX_NORMALIZED_VEL), fontsize=9, color='r')
    axs[1].set_xlim(-1, 1)

    axs[2].plot(tau, acc, 'orange', linewidth=2)
    axs[2].set_ylabel("Acceleration f''(τ)", fontsize=11)
    axs[2].grid(True, alpha=0.3)
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axs[2].axhline(y=MAX_NORMALIZED_ACC, color='r', linestyle='--', alpha=0.5)
    axs[2].axhline(y=-MAX_NORMALIZED_ACC, color='r', linestyle='--', alpha=0.5)
    axs[2].axvline(x=-0.5, color='b', linestyle='--', alpha=0.3)
    axs[2].axvline(x=0.5, color='b', linestyle='--', alpha=0.3)
    axs[2].set_xlim(-1, 1)

    axs[3].plot(tau, jerk, 'r-', linewidth=2)
    axs[3].set_ylabel("Jerk f'''(τ)", fontsize=11)
    axs[3].set_xlabel('Normalized time τ', fontsize=11)
    axs[3].grid(True, alpha=0.3)
    axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axs[3].set_xlim(-1, 1)

    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/normalized_curve.png', dpi=150, bbox_inches='tight')
    print("Saved: img/normalized_curve.png")
    plt.close()


def plot_rest_to_rest_comparison():
    """Compare rest-to-rest motion: original vs generalized."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    x0, x1 = 0, 10
    robotVmax, robotAmax = 5, 2

    # Original implementation
    T_orig = scb.motionTime(robotVmax, robotAmax, abs(x1 - x0))
    t_orig = np.linspace(0, T_orig, 501)
    pos_orig = scb.sCurve(t_orig, T_orig, x0, x1)

    # Generalized implementation
    T_gen, tau_start, tau_end = generalized_motion_time(robotVmax, robotAmax, x0, x1, 0, 0, 0, 0)
    t_gen = np.linspace(0, T_gen, 501)
    pos_gen = generalized_sCurve(t_gen, T_gen, x0, x1, tau_start, tau_end)
    vel_gen = get_velocity(t_gen, T_gen, x0, x1, tau_start, tau_end)
    acc_gen = get_acceleration(t_gen, T_gen, x0, x1, tau_start, tau_end)

    # Plot position comparison
    axs[0, 0].plot(t_orig, pos_orig, 'b-', linewidth=2, label='Original')
    axs[0, 0].plot(t_gen, pos_gen, 'r--', linewidth=2, label='Generalized')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Position')
    axs[0, 0].set_title('Position: Original vs Generalized')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot velocity
    axs[0, 1].plot(t_gen, vel_gen, 'g-', linewidth=2)
    axs[0, 1].axhline(y=robotVmax, color='r', linestyle='--', label=f'Vmax={robotVmax}')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Velocity')
    axs[0, 1].set_title('Velocity Profile')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot acceleration
    axs[1, 0].plot(t_gen, acc_gen, 'orange', linewidth=2)
    axs[1, 0].axhline(y=robotAmax, color='r', linestyle='--', label=f'Amax={robotAmax}')
    axs[1, 0].axhline(y=-robotAmax, color='r', linestyle='--')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Acceleration')
    axs[1, 0].set_title('Acceleration Profile')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Plot error
    pos_error = pos_gen - np.interp(t_gen, t_orig, pos_orig)
    axs[1, 1].plot(t_gen, pos_error * 1000, 'purple', linewidth=2)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Position Error (×10⁻³)')
    axs[1, 1].set_title('Position Difference (Original - Generalized)')
    axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Rest-to-Rest Motion: x0={x0}, x1={x1}, Vmax={robotVmax}, Amax={robotAmax}', fontsize=12)
    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/rest_to_rest_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: img/rest_to_rest_comparison.png")
    plt.close()


def plot_custom_boundary_conditions():
    """Demonstrate motions with various boundary conditions."""
    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    x0, x1 = 0, 10
    robotVmax, robotAmax = 8, 4

    # Different scenarios
    scenarios = [
        {'v0': 0, 'v1': 0, 'label': 'Rest-to-Rest\n(v₀=0, v₁=0)'},
        {'v0': 3, 'v1': 0, 'label': 'Moving Start\n(v₀=3, v₁=0)'},
        {'v0': 0, 'v1': 2, 'label': 'Moving End\n(v₀=0, v₁=2)'},
    ]

    for col, scenario in enumerate(scenarios):
        v0, v1 = scenario['v0'], scenario['v1']

        plan = plan_motion(x0, x1, v0=v0, v1=v1, robotVmax=robotVmax, robotAmax=robotAmax)
        T = plan['T']
        tau_start, tau_end = plan['tau_start'], plan['tau_end']

        t = np.linspace(0, T, 501)
        pos = generalized_sCurve(t, T, x0, x1, tau_start, tau_end)
        vel = get_velocity(t, T, x0, x1, tau_start, tau_end)
        acc = get_acceleration(t, T, x0, x1, tau_start, tau_end)

        # Position
        axs[0, col].plot(t, pos, 'b-', linewidth=2)
        axs[0, col].scatter([0, T], [pos[0], pos[-1]], color='red', s=50, zorder=5)
        axs[0, col].set_ylabel('Position' if col == 0 else '')
        axs[0, col].set_title(scenario['label'])
        axs[0, col].grid(True, alpha=0.3)

        # Velocity
        axs[1, col].plot(t, vel, 'g-', linewidth=2)
        axs[1, col].axhline(y=robotVmax, color='r', linestyle='--', alpha=0.5)
        axs[1, col].scatter([0, T], [vel[0], vel[-1]], color='red', s=50, zorder=5)
        axs[1, col].set_ylabel('Velocity' if col == 0 else '')
        axs[1, col].grid(True, alpha=0.3)
        axs[1, col].annotate(f'v₀={vel[0]:.2f}', xy=(0.1*T, vel[0]+0.3), fontsize=9)
        axs[1, col].annotate(f'v₁={vel[-1]:.2f}', xy=(T*0.6, vel[-1]+0.3), fontsize=9)

        # Acceleration
        axs[2, col].plot(t, acc, 'orange', linewidth=2)
        axs[2, col].axhline(y=robotAmax, color='r', linestyle='--', alpha=0.5)
        axs[2, col].axhline(y=-robotAmax, color='r', linestyle='--', alpha=0.5)
        axs[2, col].scatter([0, T], [acc[0], acc[-1]], color='red', s=50, zorder=5)
        axs[2, col].set_xlabel('Time (s)')
        axs[2, col].set_ylabel('Acceleration' if col == 0 else '')
        axs[2, col].grid(True, alpha=0.3)

    plt.suptitle('Generalized S-Curve: Custom Boundary Conditions', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/custom_boundary_conditions.png', dpi=150, bbox_inches='tight')
    print("Saved: img/custom_boundary_conditions.png")
    plt.close()


def plot_curve_segment_visualization():
    """Visualize which segment of the curve is used for different boundary conditions."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Full normalized curve
    tau_full = np.linspace(-1, 1, 1001)
    f_full = normalized_f(tau_full)
    f_prime_full = normalized_f_derivative(tau_full)
    f_double_prime_full = normalized_f_second_derivative(tau_full)

    # Scenarios with different boundary conditions
    scenarios = [
        {'v0': 0, 'v1': 0, 'label': 'Rest-to-Rest (full curve)', 'color': 'blue'},
        {'v0': 3, 'v1': 0, 'label': 'v₀=3, v₁=0 (partial)', 'color': 'green'},
        {'v0': 0, 'v1': 2, 'label': 'v₀=0, v₁=2 (partial)', 'color': 'orange'},
    ]

    x0, x1 = 0, 10
    robotVmax, robotAmax = 8, 4

    # Position curve with segments
    ax = axs[0, 0]
    ax.plot(tau_full, f_full, 'k-', linewidth=1, alpha=0.3, label='Full curve')

    for scenario in scenarios:
        v0, v1 = scenario['v0'], scenario['v1']
        plan = plan_motion(x0, x1, v0=v0, v1=v1, robotVmax=robotVmax, robotAmax=robotAmax)
        tau_start, tau_end = plan['tau_start'], plan['tau_end']

        # Plot this segment
        mask = (tau_full >= tau_start) & (tau_full <= tau_end)
        ax.plot(tau_full[mask], f_full[mask], linewidth=3, label=scenario['label'], color=scenario['color'])
        ax.scatter([tau_start, tau_end], [normalized_f(tau_start), normalized_f(tau_end)],
                   s=80, color=scenario['color'], zorder=5, edgecolors='black')

    ax.set_xlabel('τ (normalized time)')
    ax.set_ylabel('f(τ) (normalized position)')
    ax.set_title('Position Curve Segments')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Velocity curve with segments
    ax = axs[0, 1]
    ax.plot(tau_full, f_prime_full, 'k-', linewidth=1, alpha=0.3, label='Full curve')

    for scenario in scenarios:
        v0, v1 = scenario['v0'], scenario['v1']
        plan = plan_motion(x0, x1, v0=v0, v1=v1, robotVmax=robotVmax, robotAmax=robotAmax)
        tau_start, tau_end = plan['tau_start'], plan['tau_end']

        mask = (tau_full >= tau_start) & (tau_full <= tau_end)
        ax.plot(tau_full[mask], f_prime_full[mask], linewidth=3, label=scenario['label'], color=scenario['color'])
        ax.scatter([tau_start, tau_end],
                   [normalized_f_derivative(tau_start), normalized_f_derivative(tau_end)],
                   s=80, color=scenario['color'], zorder=5, edgecolors='black')

    ax.set_xlabel('τ (normalized time)')
    ax.set_ylabel("f'(τ) (normalized velocity)")
    ax.set_title('Velocity Curve Segments')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Phase space view
    ax = axs[1, 0]
    ax.plot(f_prime_full, f_double_prime_full, 'k-', linewidth=1, alpha=0.3)

    for scenario in scenarios:
        v0, v1 = scenario['v0'], scenario['v1']
        plan = plan_motion(x0, x1, v0=v0, v1=v1, robotVmax=robotVmax, robotAmax=robotAmax)
        tau_start, tau_end = plan['tau_start'], plan['tau_end']

        mask = (tau_full >= tau_start) & (tau_full <= tau_end)
        ax.plot(f_prime_full[mask], f_double_prime_full[mask], linewidth=3,
                label=scenario['label'], color=scenario['color'])

    ax.set_xlabel("Velocity f'(τ)")
    ax.set_ylabel("Acceleration f''(τ)")
    ax.set_title('Phase Space (Velocity vs Acceleration)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Text explanation
    ax = axs[1, 1]
    ax.axis('off')
    explanation = """
    Curve Segment Selection:

    The S-curve contains all possible motion states.
    For custom boundary conditions, we select the
    segment that matches desired start/end states:

    • τ = -1: Rest state (v=0, a=0) at start
    • τ = 0:  Maximum velocity, zero acceleration
    • τ = +1: Rest state (v=0, a=0) at end

    The relationship between velocity and acceleration
    is fixed by the curve shape. Given a velocity,
    we find the corresponding τ value, which also
    determines the acceleration at that point.

    Key insight: We're not creating new curves,
    just using different portions of the same
    optimal S-curve!
    """
    ax.text(0.1, 0.9, explanation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Understanding Curve Segment Selection', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/curve_segment_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: img/curve_segment_visualization.png")
    plt.close()


def plot_motion_continuation():
    """Demonstrate smooth motion continuation (chaining motions)."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    robotVmax, robotAmax = 6, 3

    # First motion: rest to moving (end with v=2)
    plan1 = plan_motion(0, 10, v0=0, v1=2, robotVmax=robotVmax, robotAmax=robotAmax)
    T1 = plan1['T']

    t1 = np.linspace(0, T1, 301)
    pos1, vel1, acc1 = evaluate_motion(plan1, t1)

    # Get actual end velocity from first motion
    v_end1 = plan1['v1_actual']
    a_end1 = plan1['a1_actual']

    print(f"Motion 1: 0 -> 10")
    print(f"  Duration: {T1:.3f}s")
    print(f"  End velocity (actual): {v_end1:.4f}")
    print(f"  End acceleration (actual): {a_end1:.4f}")

    # Second motion: continue from end state to rest at position 15
    plan2 = plan_motion(10, 15, v0=v_end1, v1=0, a0=a_end1, robotVmax=robotVmax, robotAmax=robotAmax)
    T2 = plan2['T']

    t2 = np.linspace(0, T2, 301)
    pos2, vel2, acc2 = evaluate_motion(plan2, t2)

    print(f"\nMotion 2: 10 -> 15 (continuing from motion 1)")
    print(f"  Duration: {T2:.3f}s")
    print(f"  Start velocity (actual): {plan2['v0_actual']:.4f}")
    print(f"  Start velocity (requested): {plan2['v0_requested']:.4f}")
    print(f"  Velocity continuity error: {abs(v_end1 - plan2['v0_actual']):.6f}")

    # Combined timeline
    t_combined = np.concatenate([t1, T1 + t2[1:]])
    pos_combined = np.concatenate([pos1, pos2[1:]])
    vel_combined = np.concatenate([vel1, vel2[1:]])
    acc_combined = np.concatenate([acc1, acc2[1:]])

    # Plot position
    axs[0].plot(t_combined, pos_combined, 'b-', linewidth=2)
    axs[0].axvline(x=T1, color='r', linestyle='--', alpha=0.5, label='Motion 1→2 transition')
    axs[0].scatter([0, T1, T1+T2], [pos1[0], pos1[-1], pos2[-1]], color='red', s=80, zorder=5)
    axs[0].set_ylabel('Position')
    axs[0].set_title('Smooth Motion Continuation: 0→10 (moving end) → 15 (rest)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].annotate('Start\n(rest)', (0, pos1[0]), textcoords="offset points", xytext=(10, 10), fontsize=9)
    axs[0].annotate('Transition\n(moving)', (T1, pos1[-1]), textcoords="offset points", xytext=(10, -20), fontsize=9)
    axs[0].annotate('End\n(rest)', (T1+T2, pos2[-1]), textcoords="offset points", xytext=(-30, 10), fontsize=9)

    # Plot velocity
    axs[1].plot(t_combined, vel_combined, 'g-', linewidth=2)
    axs[1].axvline(x=T1, color='r', linestyle='--', alpha=0.5)
    axs[1].axhline(y=robotVmax, color='orange', linestyle='--', alpha=0.5, label=f'Vmax={robotVmax}')
    axs[1].axhline(y=-robotVmax, color='orange', linestyle='--', alpha=0.5)
    axs[1].scatter([0, T1, T1+T2], [vel1[0], vel1[-1], vel2[-1]], color='red', s=80, zorder=5)
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Plot acceleration
    axs[2].plot(t_combined, acc_combined, 'orange', linewidth=2)
    axs[2].axvline(x=T1, color='r', linestyle='--', alpha=0.5)
    axs[2].axhline(y=robotAmax, color='r', linestyle='--', alpha=0.5, label=f'Amax={robotAmax}')
    axs[2].axhline(y=-robotAmax, color='r', linestyle='--', alpha=0.5)
    axs[2].scatter([0, T1, T1+T2], [acc1[0], acc1[-1], acc2[-1]], color='red', s=80, zorder=5)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/motion_continuation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: img/motion_continuation.png")
    plt.close()


if __name__ == '__main__':
    print("Generating plots for generalized S-curve motion...\n")

    plot_normalized_curve()
    plot_rest_to_rest_comparison()
    plot_custom_boundary_conditions()
    plot_curve_segment_visualization()
    plot_motion_continuation()

    print("\nAll plots generated successfully!")
