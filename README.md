[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This file is also available as `README.nb` *Mathematica* notebook that replicates all derivations and plots.

# `s-curve-beta`: Efficient Python implementation of the smoothest S-curve robot motion planner ever

1. It's 2022 and you're still using a trapezoid motion profile?
2. Your robot dog is vibrating like a washing machine?
3. Tired of the mess in the code because of the multiple piecewise time regions?
4. Want a simple single-formula smooth motion profile?
5. Want an S-curve implementation that literally fits in 20 lines of code?

If you answer "yes" to any of the above, read on.

Based on the answer of Cye Waldman (https://math.stackexchange.com/a/2403818) I present this Python package to calculate the S-curve for the robot motion planning with a given maximum velocity and acceleration with a single formula.

Here is the complete motion formula:

$$f\left( t, p \right)=\frac{1}{2}\left[ 1+\text{sgn} (x)\cdot \frac{B\left( {1}/{2}, p+1,{t^2} \right)}{B\left( {1}/{2}, p+1 \right)} \right],$$

$$motionTime(robotVmax, robotAmax,motionRange)=\\
max\left[\frac{2\cdot3^{3/4}}{\sqrt{\pi}}\cdot\sqrt\frac{motionRange}{robotAmax},\frac{32}{5\pi}\cdot\frac{motionRange}{robotVmax}\right],$$

$$position(t,robotVmax,robotAmax,motionRange)=\\
motionRange\cdot f\left(\frac{2 t}{motionTime}-1,2.5\right),$$

where *sgn(x)* is Sign function, numerator and denominator *𝐵s* are the incomplete and complete beta functions, respectively (that's where `beta` comes from in the name), *robotVmax* is the maximum velocity, *robotAmax* is the maximum acceleration and *motionRange* is the absolute value of the motion (from start to end).

Here are 2 examples of motion planning with $motionRange=2$:

![](https://github.com/hidoba/s-curve-beta/raw/main/img/plot1.png)

**The motion curve always has the same shape, while horizontal and vertical scales change.** This property has the following benefits:

1. The curve (function $f(t,2.5)$ for $-1\le t \le 1$) can be precomputed, so instead of calculating Beta functions each time it's possible to do the linear interpolation of the precomputed points. They can be programmed in a microcontroller.
2. The code is much simpler (<20 lines).
3. The movements appear more natural and human-like.

The disadvantages are:
1. **The movements may take longer time to complete for large motions.**
2. You can't simply put a linear area in the middle of the curve, that would introduce discontinuity and possibly large values in jerk.


# Installation

Using pip (recommended):
```
pip install s-curve-beta
```
From source:
```
git clone https://github.com/hidoba/s-curve-beta.git
cd s-curve-beta
python setup.py install
```

# Examples

Calculating robot position at given moments of time:
```python
import scurvebeta as scb

# motion parameters
max_velocity = 12
max_acceleration = 3
x0 = -3
x1 = 10

# Calculate motion time
motionTime = scb.motionTime(max_velocity, max_acceleration, abs(x0-x1))
print("motionTime = ", motionTime, "seconds")

# Calculate position at a given time
t = 2.3
print("position(",t,") = ", scb.sCurve(t, motionTime, x0, x1))

# Calculate multiple positions at given moments of time (faster, most recommended)
import numpy as np
t = np.array([-1,0,1,2,3,motionTime,10])
print("position(",t,") = ", scb.sCurve(t, motionTime, x0, x1))
```

Making plots:
```python
import scurvebeta as scb
import matplotlib.pyplot as plt
import numpy as np

def plotMotion(max_velocity, max_acceleration, x0, x1):
    motionTime = scb.motionTime(max_velocity, max_acceleration, abs(x0-x1))
    dt = 0.08
    t = np.arange(0,motionTime, dt)
    pos = scb.sCurve(t, motionTime, x0, x1)
    vel = np.diff(pos)/dt
    acc = np.diff(vel)/dt
    jerk = np.diff(acc)/dt

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, pos)
    axs[0].set_ylabel('Position')
    axs[0].grid(True)

    axs[1].plot(t[:-1], vel, label='velocity')
    axs[1].plot(t[:-2], acc, label='acceleration')
    axs[1].plot(t[:-3], jerk, label='jerk')
    axs[1].axhline(y = max_acceleration, color = 'orange', linestyle = '--')
    axs[1].axhline(y = max_velocity, color = 'blue', linestyle = '--')
    axs[1].text(0,max_acceleration+0.15,'max_acceleration='+str(max_acceleration))
    axs[1].text(0,max_velocity+0.15,'max_velocity='+str(max_velocity))
    axs[1].grid(True)
    axs[1].set_xlabel('Time')
    axs[1].legend(handlelength=4)

    fig.tight_layout()
    plt.show()

# Please note that if you need more accurate velocity / acceleration
# you should better use the 'true' version of the function instead of the 'interpolated' one.
# Pay attention at the jerk on the second plot, the sawteeth are because of the imprecision
# of very small changes of the third derivative. If you just need the position,
# interpolated function should be just fine.
plotMotion(6, 3, -3, 10)
plotMotion(2, 3, -3, 10)
```
![](https://github.com/hidoba/s-curve-beta/raw/main/img/pyplot1.png)
![](https://github.com/hidoba/s-curve-beta/raw/main/img/pyplot2.png)

## Interpolated vs True function

By default s-curve-beta uses an interpolated version of the *f* function (using 801 points interpolation). It's very fast if used on the arrays and it can be easily adapted for microcontrollers.

You can use the true function (requires scipy) by importing an optional `scurvebetatrue` module:

```python
from scurvebeta import scurvebetatrue
print(scurvebetatrue.sCurve_true(2.3,15,-1,5))

import scurvebeta as scb
print(scb.sCurve(2.3,15,-1,5))
```
```
-0.8849490453964555
-0.8849435436031998
```
Compare the execution speed:
```python
import timeit
from scurvebeta import scurvebetatrue
import scurvebeta as scb
import numpy as np

def trueFunction():
    return scurvebetatrue.sCurve_true(2.3,15,-1,5)

def trueFunctionArray():
    return scurvebetatrue.sCurve_true(np.arange(0,15,15/100000),15,-1,5)

def interpolatedFunction():
    return scb.sCurve(2.3,15,-1,5)

def interpolatedFunctionArray():
    return scb.sCurve(np.arange(0,15,15/100000),15,-1,5)

print(timeit.timeit('trueFunction()',number=100000, setup="from __main__ import trueFunction"))
# 0.4408080680000239

print(timeit.timeit('trueFunctionArray()',number=1, setup="from __main__ import trueFunctionArray"))
# 0.4047970910000913

print(timeit.timeit('interpolatedFunction()',number=100000, setup="from __main__ import interpolatedFunction"))
# 0.381616509999958

print(timeit.timeit('interpolatedFunctionArray()',number=1, setup="from __main__ import interpolatedFunctionArray"))
# 0.0019440340000755896
```

Plots with "True" function:

```python
import scurvebeta as scb
from scurvebeta import scurvebetatrue
import matplotlib.pyplot as plt
import numpy as np

def plotMotionTrue(max_velocity, max_acceleration, x0, x1):
    motionTime = scb.motionTime(max_velocity, max_acceleration, abs(x0-x1))
    dt = 0.005
    t = np.arange(0,motionTime, dt)
    pos = scurvebetatrue.sCurve_true(t, motionTime, x0, x1)
    vel = np.diff(pos)/dt
    acc = np.diff(vel)/dt
    jerk = np.diff(acc)/dt

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, pos)
    axs[0].set_ylabel('Position')
    axs[0].grid(True)

    axs[1].plot(t[:-1], vel, label='velocity')
    axs[1].plot(t[:-2], acc, label='acceleration')
    axs[1].plot(t[:-3], jerk, label='jerk')
    axs[1].axhline(y = max_acceleration, color = 'orange', linestyle = '--')
    axs[1].axhline(y = max_velocity, color = 'blue', linestyle = '--')
    axs[1].text(0,max_acceleration+0.15,'max_acceleration='+str(max_acceleration))
    axs[1].text(0,max_velocity+0.15,'max_velocity='+str(max_velocity))
    axs[1].grid(True)
    axs[1].set_xlabel('Time')
    axs[1].legend(handlelength=4)

    fig.tight_layout()
    plt.show()

plotMotionTrue(6, 3, -3, 10)
plotMotionTrue(2, 3, -3, 10)
```
![](https://github.com/hidoba/s-curve-beta/raw/main/img/pyplot3.png)
![](https://github.com/hidoba/s-curve-beta/raw/main/img/pyplot4.png)

# Multiple axes synchronization

To synchronize multiple axes simply calculate the maximum motion time of all axes and use it for every axis. This will minimize the maximum jerk and acceleration of the initially faster axes, decreasing the vibrations of the robot.

# Derivations

Below derivations include *Mathematica* code that can be used to replicate them.

This file is also available as `README.nb` *Mathematica* notebook that replicates all derivations and plots.

The original *f* function comes from the integration of the [superparabola](https://en.wikipedia.org/wiki/Superparabola), [done](https://math.stackexchange.com/a/2403818) by Cye Waldman. We can prove that his integration is correct:

```mathematica
f[x_, p_] := 
 1/2 (1 + RealSign[x]*Beta[x^2, 1/2, p + 1]/Beta[1/2, p + 1])

FullSimplify[
 Integrate[(1 - x^2)^p/Beta[1/2, 1 + p], {x, -1, t}] - f[t, p], 
 Element[p, Reals] && Element[t, Reals] && 0 <= t < 1 && p > -1]

 (* 0 *)
```
## Proof that the most optimal $p$ is 2.5

My suggested value of $p$ is 2.5 which gives the lowest possible absolute jerk, let's see why.

Visualizing derivatives with different values of $p$:

```mathematica
Animate[
 Show[
  Plot[Evaluate@Table[D[f[x, 2.5], {x, i}], {i, 0, 3}], {x, -1, 1}, 
   PlotLegends -> {"position f(t,2.5)", "velocity f'(t,2.5)", 
     "acceleration f''(t,2.5)", "jerk f'''(t,2.5)"}, 
   PlotRange -> {-11, 8}, PlotLabel -> "p = " ~~ ToString[param]],
  Plot[Evaluate@Table[D[f[x, param], {x, i}], {i, 0, 3}], {x, -1, 1}, 
   PlotLegends -> (StringJoin[#, 
        "(t," ~~ ToString[param] ~~ ")"] & /@ {"position f", 
       "velocity f'", "acceleration f''", "jerk f'''"}), 
   PlotRange -> All, PlotStyle -> Dashed]
  ],
 {param, 2.00001, 5, 0.15}]
```

![](https://github.com/hidoba/s-curve-beta/raw/main/img/plot2.gif)

Calculating largest jerks for different values of the parameter $p$:

```mathematica
jerkValues[param_] := {Limit[D[f[x, param], {x, 3}], x -> 0],
  Maximize[{D[f[x, param], {x, 3}], -1 <= x <= 1}, x][[1]]}}

paramData = Table[{p, jerkValues[p]}, {p, 2.001, 3, 0.01}];

ListLinePlot[
 Abs[Transpose[(Outer[List, {#1}, #2][[1]] & @@@ paramData)]], 
 PlotLegends -> {"|max negative jerk| (at x=0)", "max positive jerk"},
  Epilog -> {Dashed, InfiniteLine[{2.5, 0}, {0, 1}]}, 
 AxesLabel -> {"p", "max jerk"}]
```

![](https://github.com/hidoba/s-curve-beta/raw/main/img/plot3.png)

The best value of the parameter $p$ happens to be 2.5, where the max absolute jerk is the smallest.

### Discussion on $2\le p<2.5$

It's possible to achieve **slightly lower maximum robot accelerations** at $2\le p<2.5$:

```mathematica
param = 2;
Show[
  Plot[Evaluate@Table[D[f[x, 2.5], {x, i}], {i, 0, 3}], {x, -1, 1}, PlotLegends -> {"position f(t,2.5)", "velocity f'(t,2.5)", "acceleration f''(t,2.5)", "jerk f'''(t,2.5)"}, PlotRange -> All], 
  Plot[Evaluate@Table[D[f[x, param], {x, i}], {i, 0, 3}], {x, -1, 1}, PlotLegends -> (StringJoin[#, "(t," ~~ ToString[param] ~~ ")"] & /@ {"position f", "velocity f'", "acceleration f''", "jerk f'''"}), PlotRange -> All, PlotStyle -> Dashed] 
 ]
```

![](https://github.com/hidoba/s-curve-beta/raw/main/img/plot4.png)

Values of $2\le p<2.5$ can shorten the robot motion time a little bit at the cost of increased maximum jerk hence increased robot vibrations at the beginning and at the end of the motion. I have decided not to implement this at the moment.

## Rescaling function *f* for the particular motion parameters

### 1. Motion time from the maximum velocity constraint

Max velocity at $t=0$:

```mathematica
maxVelocity = Limit[D[f[x, 5/2], x], x -> 0]
```

$$\frac{16}{5 \pi}$$

With the given maximum robot velocity *robotVmax* the shortest motion time can be derived from the equation:

$$robotVmax=\frac{2\frac{16}{5\pi}motionRange}{time}$$

Hence,

$$time = \frac{32}{5 \pi}\cdot\frac{motionRange}{robotVmax}$$

### 2. Motion time from the maximum acceleration constraint

Calculate max acceleration at $p=2.5$:

```mathematica
FindMaximum[{D[f[x, 5/2], {x, 2}], -1 < x < 0}, x]

(*{1.65399, {x -> -0.5}}*)
```

Maximum acceleration happens to be at $t=-\frac{1}{2}$

```mathematica
D[f[x, 5/2], {x, 2}] /. x -> -1/2
```

$$\frac{3\sqrt3}{\pi}$$

With the given maximum robot acceleration *robotAmax* the shortest motion time can be derived from the equation:

$$\text{robotAmax}=\frac{4\frac{3\sqrt{3}}{\pi }\text{motionRange}}{time^2}$$

Hence,

$$time=\frac{2\cdot3^{3/4}}{\sqrt{\pi}}\cdot\sqrt\frac{motionRange}{robotAmax}$$

### 3. Motion time with both constraints

To consider both maximum velocity and maximum acceleration constraints we have to take the maximum of the above motion times:

$$motionTime=
max\left[\frac{2\cdot3^{3/4}}{\sqrt{\pi}}\cdot\sqrt\frac{motionRange}{robotAmax},\frac{32}{5\pi}\cdot\frac{motionRange}{robotVmax}\right],$$

In the future I may add a maximum jerk constraint.

### 4. Final motion position formula

We have to rescale *t* in $f(t,2.5)$ in such a way that the motion would start at $t=0$ and end at $t=motionTime$. Additionally we have to rescale the value of *f* to go from 0 to *motionRange*. After rescaling we get:

$$position(t,robotAmax,motionRange)=\\
motionRange\cdot f \left(\frac{2 t}{motionTime}-1,2.5 \right)$$

## Example robot motions limited by acceleration

![](https://github.com/hidoba/s-curve-beta/raw/main/img/plot5.png)

# Generalized Motion States (Custom Boundary Conditions)

The original S-curve implementation assumes motion starts and ends at rest (zero velocity and acceleration). The **generalized motion module** extends this to support arbitrary initial and final states (position, velocity, acceleration).

## Key Insight

The S-curve already contains **all possible motion states** along its path. Instead of creating new curves, we simply select the segment of the existing curve that matches our desired boundary conditions:

- **τ = -1**: Rest state (v=0, a=0) at start of curve
- **τ = 0**: Maximum velocity, zero acceleration (midpoint)
- **τ = +1**: Rest state (v=0, a=0) at end of curve

For any intermediate τ value, we get a unique (position, velocity, acceleration) state. The velocity and acceleration are coupled by the curve shape.

![](https://github.com/hidoba/s-curve-beta/raw/main/img/curve_segment_visualization.png)

## Installation

The generalized module requires `scipy` in addition to `numpy`:

```bash
pip install scipy
```

## Basic Usage

```python
from scurvebeta.generalized import plan_motion, evaluate_motion
import numpy as np

# Plan a motion from position 0 to 10
# Starting with velocity 3, ending at rest
plan = plan_motion(
    x0=0, x1=10,           # Start and end positions
    v0=3, v1=0,            # Start and end velocities
    robotVmax=8,           # Maximum velocity constraint
    robotAmax=4            # Maximum acceleration constraint
)

print(f"Motion time: {plan['T']:.3f} seconds")
print(f"Actual start velocity: {plan['v0_actual']:.3f}")
print(f"Actual end velocity: {plan['v1_actual']:.3f}")

# Evaluate position, velocity, acceleration at any time
t = np.linspace(0, plan['T'], 100)
position, velocity, acceleration = evaluate_motion(plan, t)
```

## Motion Continuation (Chaining Motions)

One powerful use case is smoothly chaining multiple motions. The end state of one motion becomes the start state of the next:

```python
from scurvebeta.generalized import plan_motion, evaluate_motion

robotVmax, robotAmax = 6, 3

# First motion: 0 → 10, start at rest, end moving
plan1 = plan_motion(0, 10, v0=0, v1=2, robotVmax=robotVmax, robotAmax=robotAmax)

# Get the actual end state
v_end = plan1['v1_actual']
a_end = plan1['a1_actual']

# Second motion: 10 → 15, continue from previous state, end at rest
plan2 = plan_motion(10, 15, v0=v_end, v1=0, a0=a_end, robotVmax=robotVmax, robotAmax=robotAmax)

# The velocity is continuous at the transition!
print(f"Motion 1 end velocity: {plan1['v1_actual']:.4f}")
print(f"Motion 2 start velocity: {plan2['v0_actual']:.4f}")
# Output: Both are 2.0000 - perfect continuity!
```

![](https://github.com/hidoba/s-curve-beta/raw/main/img/motion_continuation.png)

## Comparison: Rest-to-Rest vs Custom Boundaries

The generalized module produces identical results to the original for rest-to-rest motion:

![](https://github.com/hidoba/s-curve-beta/raw/main/img/rest_to_rest_comparison.png)

But it can also handle cases where motion starts or ends while moving:

![](https://github.com/hidoba/s-curve-beta/raw/main/img/custom_boundary_conditions.png)

## API Reference

### `plan_motion(x0, x1, v0=0, v1=0, a0=0, a1=0, robotVmax=None, robotAmax=None)`

Plan a motion from state (x0, v0, a0) to state (x1, v1, a1).

**Parameters:**
- `x0, x1`: Start and end positions
- `v0, v1`: Start and end velocities (default 0 for rest)
- `a0, a1`: Start and end accelerations (default 0)
- `robotVmax`: Maximum velocity constraint (optional)
- `robotAmax`: Maximum acceleration constraint (optional)

**Returns:** Dictionary with:
- `T`: Motion duration
- `tau_start`, `tau_end`: Segment endpoints on normalized curve
- `v0_actual`, `v1_actual`: Actual boundary velocities achieved
- `a0_actual`, `a1_actual`: Actual boundary accelerations achieved

### `evaluate_motion(plan, t)`

Evaluate the motion at time(s) t.

**Returns:** `(position, velocity, acceleration)` tuple

### Lower-level Functions

```python
from scurvebeta.generalized import (
    generalized_motion_time,  # Calculate T, tau_start, tau_end
    generalized_sCurve,       # Get position at time t
    get_velocity,             # Get velocity at time t
    get_acceleration,         # Get acceleration at time t
    normalized_f,             # Normalized position function f(τ)
    normalized_f_derivative,  # Velocity profile f'(τ)
    normalized_f_second_derivative,  # Acceleration profile f''(τ)
)
```

## Mathematical Foundation

The normalized S-curve f(τ) for τ ∈ [-1, 1] has these properties:

- **Position**: f(τ) ranges from 0 to 1
- **Velocity**: f'(τ) = (1 - τ²)^2.5 / B(0.5, 3.5), max ≈ 1.019 at τ=0
- **Acceleration**: f''(τ) = -5τ(1 - τ²)^1.5 / B(0.5, 3.5), max ≈ 1.654 at τ=-0.5

When using segment [τ_start, τ_end] mapped to time [0, T] and position [x0, x1]:

$$position(t) = x_0 + (x_1 - x_0) \cdot \frac{f(\tau(t)) - f(\tau_{start})}{f(\tau_{end}) - f(\tau_{start})}$$

$$velocity(t) = \frac{(x_1 - x_0) \cdot \Delta\tau}{\Delta f \cdot T} \cdot f'(\tau(t))$$

$$acceleration(t) = \frac{(x_1 - x_0) \cdot \Delta\tau^2}{\Delta f \cdot T^2} \cdot f''(\tau(t))$$

where Δτ = τ_end - τ_start and Δf = f(τ_end) - f(τ_start).

## Limitations

1. **Coupled velocity and acceleration**: At any point on the curve, velocity and acceleration are linked. You cannot specify arbitrary (v, a) pairs independently.

2. **Velocity direction**: The start/end velocity should be in the same direction as the motion (from x0 to x1). Opposite-direction velocities are not fully supported.

3. **Numerical search**: For custom boundary conditions, the algorithm uses numerical search to find the optimal curve segment, which may be slower than the closed-form rest-to-rest solution.

## License

Copyright (c) 2022 Vladimir Grankovsky at Hidoba Research. This work is licensed under an Apache 2.0 license.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.