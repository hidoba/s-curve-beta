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
    """Demonstrate PERFECT C∞ smooth motion continuation using nested beta curves."""
    from scurvebeta.generalized import continue_motion, evaluate_smooth_motion

    fig, axs = plt.subplots(9, 1, figsize=(14, 22))

    robotVmax, robotAmax = 6, 3

    # First motion: rest to moving (end mid-curve)
    plan1 = plan_motion(0, 10, v0=0, v1=2, robotVmax=robotVmax, robotAmax=robotAmax)
    T1 = plan1['T']

    t1 = np.linspace(0, T1, 301)
    pos1, vel1, acc1 = evaluate_motion(plan1, t1)

    print(f"Motion 1: 0 -> 10")
    print(f"  Duration: {T1:.3f}s")
    print(f"  End: v={plan1['v1_actual']:.4f}, a={plan1['a1_actual']:.4f}")

    # Continue to ARBITRARY target with PERFECT continuity using smooth τ(t)
    x_target = 18  # User's choice - not constrained by physics!
    plan2 = continue_motion(plan1, x_target=x_target, robotVmax=robotVmax, robotAmax=robotAmax)
    T2 = plan2['T']

    t2 = np.linspace(0, T2, 301)
    pos2, vel2, acc2 = evaluate_smooth_motion(plan2, t2)
    motion2 = plan2['smooth_motion']

    print(f"\nMotion 2 (smooth continuation): {plan2['x0']:.2f} -> {plan2['x1']:.2f}")
    print(f"  Duration: {T2:.3f}s")
    print(f"  Start: v={plan2['v0_actual']:.4f}, a={plan2['a0_actual']:.4f}")

    # Check continuity
    v_jump = abs(plan2['v0_actual'] - plan1['v1_actual'])
    a_jump = abs(plan2['a0_actual'] - plan1['a1_actual'])
    print(f"\nCONTINUITY CHECK:")
    print(f"  Velocity jump: {v_jump:.12f}")
    print(f"  Acceleration jump: {a_jump:.12f}")
    print(f"  PERFECT CONTINUITY: {v_jump < 1e-6 and a_jump < 1e-6}")

    # Combined timeline
    t_combined = np.concatenate([t1, T1 + t2[1:]])
    pos_combined = np.concatenate([pos1, pos2[1:]])
    vel_combined = np.concatenate([vel1, vel2[1:]])
    acc_combined = np.concatenate([acc1, acc2[1:]])

    # Compute derivatives for motion 2
    jerk2 = motion2.jerk(t2)
    snap2 = motion2.snap(t2)
    crackle2 = motion2.crackle(t2)
    pop2 = motion2.pop(t2)
    lock2 = motion2.lock(t2)
    drop2 = motion2.drop(t2)

    # Compute derivatives numerically for motion 1
    dt1 = t1[1] - t1[0]
    jerk1 = np.gradient(acc1, dt1)
    snap1 = np.gradient(jerk1, dt1)
    crackle1 = np.gradient(snap1, dt1)
    pop1 = np.gradient(crackle1, dt1)
    lock1 = np.gradient(pop1, dt1)
    drop1 = np.gradient(lock1, dt1)

    # Combined arrays
    jerk_combined = np.concatenate([jerk1, jerk2[1:]])
    snap_combined = np.concatenate([snap1, snap2[1:]])
    crackle_combined = np.concatenate([crackle1, crackle2[1:]])
    pop_combined = np.concatenate([pop1, pop2[1:]])
    lock_combined = np.concatenate([lock1, lock2[1:]])
    drop_combined = np.concatenate([drop1, drop2[1:]])

    derivative_names = ['Position', 'Velocity', 'Acceleration', 'Jerk',
                        'Snap (4th)', 'Crackle (5th)', 'Pop (6th)', 'Lock (7th)', 'Drop (8th)']
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'magenta', 'cyan', 'olive']
    data = [pos_combined, vel_combined, acc_combined, jerk_combined,
            snap_combined, crackle_combined, pop_combined, lock_combined, drop_combined]

    for i, (name, color, d) in enumerate(zip(derivative_names, colors, data)):
        axs[i].plot(t_combined, d, color=color, linewidth=2)
        axs[i].axvline(x=T1, color='r', linestyle='--', alpha=0.5)
        axs[i].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axs[i].set_ylabel(name, fontsize=10)
        axs[i].grid(True, alpha=0.3)

        # Annotate key features
        if i == 0:
            axs[i].scatter([0, T1, T1+T2], [pos1[0], pos1[-1], pos2[-1]], color='red', s=60, zorder=5)
            axs[i].annotate('Start', (0, pos1[0]), textcoords="offset points", xytext=(10, 10), fontsize=8)
            axs[i].annotate(f'x={x_target}', (T1+T2, pos2[-1]), textcoords="offset points", xytext=(-40, 10), fontsize=8)
        elif i < 4:
            axs[i].annotate('Continuous!', (T1, d[len(t1)]),
                            textcoords="offset points", xytext=(15, 5), fontsize=9, color='green', fontweight='bold')

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    axs[-1].annotate('ALL derivatives → 0 smoothly!', (T1 + T2 - 1, 0),
                     textcoords="offset points", xytext=(-100, 20), fontsize=10, color='green', fontweight='bold')

    plt.suptitle('C∞ Smooth Motion: 9 Derivatives (Position through 8th derivative)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/motion_continuation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: img/motion_continuation.png")
    plt.close()


def plot_direction_reversal():
    """Demonstrate smooth direction reversal using nested beta curves."""
    from scurvebeta.generalized import continue_motion, evaluate_smooth_motion

    fig, axs = plt.subplots(9, 1, figsize=(14, 22))

    robotVmax, robotAmax = 6, 3

    # Start moving right
    plan1 = plan_motion(0, 8, v0=0, v1=2.5, robotVmax=robotVmax, robotAmax=robotAmax)
    T1 = plan1['T']

    print("=== Direction Reversal with Perfect Continuity ===")
    print(f"Initial motion: 0 -> 8 (ending with v={plan1['v1_actual']:.2f})")

    # Want to go BACK to x=3 (to the LEFT of where we'll end up!)
    x_target = 3
    plan2 = continue_motion(plan1, x_target=x_target, robotVmax=robotVmax, robotAmax=robotAmax)
    T2 = plan2['T']
    motion2 = plan2['smooth_motion']

    print(f"\nSmooth continuation to x={x_target}:")
    print(f"  Motion 2: {plan2['x0']:.2f} -> {plan2['x1']:.2f}, T={T2:.3f}s")

    # Check continuity
    v_jump = abs(plan2['v0_actual'] - plan1['v1_actual'])
    a_jump = abs(plan2['a0_actual'] - plan1['a1_actual'])
    print(f"  Velocity jump: {v_jump:.10f}")
    print(f"  Acceleration jump: {a_jump:.10f}")

    # Evaluate
    t1 = np.linspace(0, T1, 200)
    t2 = np.linspace(0, T2, 200)

    pos1, vel1, acc1 = evaluate_motion(plan1, t1)
    pos2, vel2, acc2 = evaluate_smooth_motion(plan2, t2)

    # Compute derivatives for motion 2
    jerk2 = motion2.jerk(t2)
    snap2 = motion2.snap(t2)
    crackle2 = motion2.crackle(t2)
    pop2 = motion2.pop(t2)
    lock2 = motion2.lock(t2)
    drop2 = motion2.drop(t2)

    # Compute derivatives numerically for motion 1
    dt1 = t1[1] - t1[0]
    jerk1 = np.gradient(acc1, dt1)
    snap1 = np.gradient(jerk1, dt1)
    crackle1 = np.gradient(snap1, dt1)
    pop1 = np.gradient(crackle1, dt1)
    lock1 = np.gradient(pop1, dt1)
    drop1 = np.gradient(lock1, dt1)

    # Combined
    t_combined = np.concatenate([t1, T1 + t2[1:]])
    pos_combined = np.concatenate([pos1, pos2[1:]])
    vel_combined = np.concatenate([vel1, vel2[1:]])
    acc_combined = np.concatenate([acc1, acc2[1:]])
    jerk_combined = np.concatenate([jerk1, jerk2[1:]])
    snap_combined = np.concatenate([snap1, snap2[1:]])
    crackle_combined = np.concatenate([crackle1, crackle2[1:]])
    pop_combined = np.concatenate([pop1, pop2[1:]])
    lock_combined = np.concatenate([lock1, lock2[1:]])
    drop_combined = np.concatenate([drop1, drop2[1:]])

    derivative_names = ['Position', 'Velocity', 'Acceleration', 'Jerk',
                        'Snap (4th)', 'Crackle (5th)', 'Pop (6th)', 'Lock (7th)', 'Drop (8th)']
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'magenta', 'cyan', 'olive']
    data = [pos_combined, vel_combined, acc_combined, jerk_combined,
            snap_combined, crackle_combined, pop_combined, lock_combined, drop_combined]

    for i, (name, color, d) in enumerate(zip(derivative_names, colors, data)):
        axs[i].plot(t_combined, d, color=color, linewidth=2)
        axs[i].axvline(x=T1, color='r', linestyle='--', alpha=0.5)
        axs[i].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axs[i].set_ylabel(name, fontsize=10)
        axs[i].grid(True, alpha=0.3)

        # Annotate key features
        if i == 0:
            axs[i].axhline(y=x_target, color='green', linestyle='--', alpha=0.5, label=f'Target x={x_target}')
            axs[i].scatter([0, T1, T1+T2], [pos1[0], pos1[-1], pos2[-1]], color='red', s=60, zorder=5)
            axs[i].legend(fontsize=9)
        elif i == 1:
            axs[i].axhline(y=robotVmax, color='orange', linestyle='--', alpha=0.3)
            axs[i].axhline(y=-robotVmax, color='orange', linestyle='--', alpha=0.3)
            axs[i].annotate('Smooth reversal', (T1 + T2*0.4, min(vel2)*0.8),
                            textcoords="offset points", xytext=(0, 0), fontsize=9, color='green')

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    axs[-1].annotate('ALL derivatives → 0 smoothly!', (T1 + T2 - 0.5, 0),
                     textcoords="offset points", xytext=(-100, 20), fontsize=10, color='green', fontweight='bold')

    plt.suptitle('Direction Reversal with C∞ Smooth Beta Curves (9 derivatives)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/user/s-curve-beta/img/direction_reversal.png', dpi=150, bbox_inches='tight')
    print("\nSaved: img/direction_reversal.png")
    plt.close()


if __name__ == '__main__':
    print("Generating plots for generalized S-curve motion...\n")

    plot_normalized_curve()
    plot_rest_to_rest_comparison()
    plot_custom_boundary_conditions()
    plot_curve_segment_visualization()
    plot_motion_continuation()
    plot_direction_reversal()

    print("\nAll plots generated successfully!")
