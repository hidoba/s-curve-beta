"""
Analyze the S-curve to understand velocity and acceleration profiles.
This will help us understand how to match arbitrary boundary conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betainc, beta

def f_true(t):
    """True beta function S-curve (normalized, from -1 to 1)"""
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t, dtype=float)
        result[t <= -1] = 0.0
        result[t >= 1] = 1.0
        mask = (t > -1) & (t < 1)
        t_masked = t[mask]
        coef = 0.5092958178940651
        neg_mask = t_masked < 0
        pos_mask = t_masked > 0
        zero_mask = t_masked == 0

        result_masked = np.zeros_like(t_masked)
        if np.any(neg_mask):
            result_masked[neg_mask] = 0.5 - coef * betainc(0.5, 3.5, t_masked[neg_mask]**2) * beta(0.5, 3.5)
        if np.any(pos_mask):
            result_masked[pos_mask] = 0.5 + coef * betainc(0.5, 3.5, t_masked[pos_mask]**2) * beta(0.5, 3.5)
        if np.any(zero_mask):
            result_masked[zero_mask] = 0.5
        result[mask] = result_masked
        return result
    else:
        if t <= -1:
            return 0.0
        if t >= 1:
            return 1.0
        coef = 0.5092958178940651
        if t < 0:
            return 0.5 - coef * betainc(0.5, 3.5, t**2) * beta(0.5, 3.5)
        if t > 0:
            return 0.5 + coef * betainc(0.5, 3.5, t**2) * beta(0.5, 3.5)
        return 0.5

# Create fine time array
t = np.linspace(-1, 1, 10001)
dt = t[1] - t[0]

# Calculate position
pos = f_true(t)

# Calculate derivatives numerically
vel = np.gradient(pos, dt)
acc = np.gradient(vel, dt)
jerk = np.gradient(acc, dt)

# Find key points
max_vel_idx = np.argmax(vel)
max_acc_idx = np.argmax(acc)
min_acc_idx = np.argmin(acc)

print("=== S-Curve Analysis ===")
print(f"\nNormalized curve from t=-1 to t=1:")
print(f"  Position range: {pos[0]:.6f} to {pos[-1]:.6f}")
print(f"  Max velocity: {vel[max_vel_idx]:.6f} at t={t[max_vel_idx]:.4f}")
print(f"  Max acceleration: {acc[max_acc_idx]:.6f} at t={t[max_acc_idx]:.4f}")
print(f"  Min acceleration: {acc[min_acc_idx]:.6f} at t={t[min_acc_idx]:.4f}")

print(f"\nTheoretical values (p=2.5):")
print(f"  Max velocity = 16/(5*pi) = {16/(5*np.pi):.6f}")
print(f"  Max acceleration = 3*sqrt(3)/pi = {3*np.sqrt(3)/np.pi:.6f}")

print(f"\nBoundary conditions at t=-1:")
print(f"  pos={pos[0]:.6f}, vel={vel[0]:.6f}, acc={acc[0]:.6f}")
print(f"\nBoundary conditions at t=0:")
print(f"  pos={pos[len(t)//2]:.6f}, vel={vel[len(t)//2]:.6f}, acc={acc[len(t)//2]:.6f}")
print(f"\nBoundary conditions at t=1:")
print(f"  pos={pos[-1]:.6f}, vel={vel[-1]:.6f}, acc={acc[-1]:.6f}")

# Plot
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(t, pos, 'b-', linewidth=2)
axs[0].set_ylabel('Position f(t)')
axs[0].set_title('Normalized S-Curve (Beta Function, p=2.5)')
axs[0].grid(True)
axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axs[0].axhline(y=1, color='k', linestyle='-', linewidth=0.5)

axs[1].plot(t, vel, 'g-', linewidth=2)
axs[1].set_ylabel("Velocity f'(t)")
axs[1].grid(True)
axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axs[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='max velocity at t=0')
axs[1].legend()

axs[2].plot(t, acc, 'orange', linewidth=2)
axs[2].set_ylabel("Acceleration f''(t)")
axs[2].grid(True)
axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axs[2].axvline(x=-0.5, color='r', linestyle='--', alpha=0.5, label='max acc at t=-0.5')
axs[2].axvline(x=0.5, color='b', linestyle='--', alpha=0.5, label='min acc at t=0.5')
axs[2].legend()

axs[3].plot(t, jerk, 'r-', linewidth=2)
axs[3].set_ylabel("Jerk f'''(t)")
axs[3].set_xlabel('Normalized time t')
axs[3].grid(True)
axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/user/s-curve-beta/img/curve_analysis.png', dpi=150)
print("\nSaved plot to img/curve_analysis.png")

# Now analyze the relationship between velocity and acceleration
# For any point on the curve, we have a unique (vel, acc) pair
print("\n=== Velocity-Acceleration Phase Space ===")
print("At different points on the curve:")
for t_val in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
    idx = int((t_val + 1) / 2 * (len(t) - 1))
    print(f"  t={t_val:+.2f}: pos={pos[idx]:.4f}, vel={vel[idx]:.4f}, acc={acc[idx]:.4f}")

# Plot phase space
fig2, ax = plt.subplots(figsize=(8, 6))
ax.plot(vel, acc, 'b-', linewidth=2)
ax.scatter([vel[0], vel[-1]], [acc[0], acc[-1]], color='green', s=100, zorder=5, label='Start/End (rest)')
ax.scatter([vel[len(t)//2]], [acc[len(t)//2]], color='red', s=100, zorder=5, label='Mid-point (max vel)')
ax.set_xlabel("Velocity f'(t)")
ax.set_ylabel("Acceleration f''(t)")
ax.set_title('S-Curve Phase Space (Velocity vs Acceleration)')
ax.grid(True)
ax.legend()
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Add arrows to show direction
for i in range(0, len(t)-1, len(t)//10):
    ax.annotate('', xy=(vel[i+100], acc[i+100]), xytext=(vel[i], acc[i]),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/user/s-curve-beta/img/phase_space.png', dpi=150)
print("Saved phase space plot to img/phase_space.png")

plt.show()
