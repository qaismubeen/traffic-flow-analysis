# MT-1008 Multivariable Calculus Semester Project
# Topic: Traffic Flow Dynamics

# We made a dataset of 30 days of traffic data with 3 variables
# time, density and speed. Then I used these to model traffic flow
# using a linear regression formula F = f(t, d, v).
# After that I found max and min flow, computed the gradient,
# directional derivative, triple integral, and compared 3 cities.

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import integrate
import os

# b = base directory (folder where this script is saved)
b = os.path.dirname(os.path.abspath(__file__))

np.random.seed(47)

# Step 1: I am creating the dataset here

# t = time (hour of day)
t = np.array([7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
              7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

# d = density (vehicles/km)
d = np.array([42,68,85,74,55,48,52,61,70,83,91,88,65,44,30,
              38,72,80,69,50,45,58,64,76,88,94,82,60,40,28])

# v = speed (km/h)
v = np.array([65,48,32,40,58,70,66,55,45,35,28,31,50,72,85,
              70,44,35,42,62,73,60,52,40,30,25,34,55,75,88])

# n = noise (small random values added to make data look realistic)
n = np.random.randint(-80, 80, 30)

# f = flow (traffic flow in vehicles per hour, calculated as density x speed plus noise)
f = d * v + n

# fr = dataframe (storing all 30 days of data in a table)
fr = pd.DataFrame({'Day': range(1,31), 't':t, 'd':d, 'v':v, 'F':f})
print(fr.to_string(index=False))

# Step 2: Saving the dataset as a csv file in the same folder as this script
fr.to_csv(os.path.join(b, 'traffic_dataset.csv'), index=False)

# Step 3: Building the regression model F = c0 + c1*t + c2*d + c3*v

# X = inputs (the three variables I am using to predict flow)
X = fr[['t','d','v']].values

# y = output (the actual flow values I want to predict)
y = fr['F'].values

# m = model (linear regression model)
m = LinearRegression()
m.fit(X, y)

# c0 = intercept (the base value in the formula when all variables are 0)
c0 = round(m.intercept_, 3)

# c1, c2, c3 = coefficients (how much each variable affects flow)
c1, c2, c3 = [round(x, 3) for x in m.coef_]
print(f"\nModel: F = {c0} + {c1}*t + {c2}*d + {c3}*v")
print(f"R² = {m.score(X,y):.4f}")

# Step 4: Finding the day with highest and lowest traffic flow

# mx = max row (day with maximum traffic flow)
mx = fr.loc[fr['F'].idxmax()]

# mn = min row (day with minimum traffic flow)
mn = fr.loc[fr['F'].idxmin()]
print(f"\nMax Flow: Day {mx['Day']}, F={mx['F']}, t={mx['t']}, d={mx['d']}, v={mx['v']}")
print(f"Min Flow: Day {mn['Day']}, F={mn['F']}, t={mn['t']}, d={mn['d']}, v={mn['v']}")

# Step 5: Computing the gradient vector at a specific point

# t0, d0, v0 = reference point (the point where I am evaluating the gradient)
t0, d0, v0 = 10, 74, 40

# gr = gradient (partial derivatives showing how flow changes with each variable)
gr = np.array([c1, c2, c3])
print(f"\nGradient ∇F at ({t0},{d0},{v0}) = {gr}")
print(f"Steepest ascent direction: {gr}")
print(f"Steepest descent direction: {-gr}")

# Step 6: Computing the directional derivative
# I am choosing a direction where density increases and speed decreases
# this represents a congestion scenario

# ur = raw direction vector (before normalizing)
ur = np.array([0, 1, -1])

# u = unit vector (normalized direction vector)
u = ur / np.linalg.norm(ur)

# dd = directional derivative (rate of change of flow in direction u)
dd = np.dot(gr, u)
print(f"\nDirectional Derivative D_u F = {dd:.4f}")

# Step 7: Computing the triple integral numerically
# Region I chose: t in [6,10], d in [20,80], v in [30,80]
# This gives total accumulated flow over that region

def F_func(vv, dv, tv):
    # vv = v value, dv = d value, tv = t value (integration variables)
    return c0 + c1*tv + c2*dv + c3*vv

# ti = triple integral result
# te = error estimate from the numerical integration
ti, te = integrate.tplquad(F_func, 6, 10, 20, 80, 30, 80)
print(f"\nTriple Integral over region = {ti:.2f} (error est: {te:.4f})")

# Step 8: Comparing average traffic flow across 3 cities

# ct = cities dictionary (storing density and speed data for each city)
ct = {
    'Karachi':  {'d': np.array([75,88,92,65,50]), 'v': np.array([35,28,25,45,60])},
    'Lahore':   {'d': np.array([60,72,80,55,42]), 'v': np.array([48,38,32,55,70])},
    'Islamabad':{'d': np.array([30,40,48,35,28]), 'v': np.array([75,65,58,80,90])},
}
print("\nCity Comparison:")
for c, vals in ct.items():
    # af = average flow (mean traffic flow for that city)
    af = np.mean(vals['d'] * vals['v'])
    print(f"  {c}: Avg Flow = {af:.1f} veh/hr")

# Step 9: Plotting the graphs

# fg = figure, ax = axes (the two side by side plots)
fg, ax = plt.subplots(1, 2, figsize=(12, 5))

# left graph shows flow vs density
ax[0].scatter(d, f, color='black', s=40)
ax[0].set_xlabel('Density (veh/km)')
ax[0].set_ylabel('Traffic Flow (veh/hr)')
ax[0].set_title('Flow vs Density')
ax[0].grid(True, linestyle='--', alpha=0.5)

# right graph shows flow vs speed
ax[1].scatter(v, f, color='black', s=40, marker='^')
ax[1].set_xlabel('Speed (km/h)')
ax[1].set_ylabel('Traffic Flow (veh/hr)')
ax[1].set_title('Flow vs Speed')
ax[1].grid(True, linestyle='--', alpha=0.5)

fg.tight_layout()

# gp = graph path (full path where the graph image will be saved)
gp = os.path.join(b, 'graphs.png')
fg.savefig(gp, dpi=150, bbox_inches='tight')

print("\nSaved at:", gp)

plt.show()
print("\nGraphs saved.")