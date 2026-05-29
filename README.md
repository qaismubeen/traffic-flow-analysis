Traffic Flow Dynamics — Multivariable Modeling and Analysis

This project applies multivariable mathematical modeling to analyze urban traffic behavior using a 30-day dataset covering three Pakistani cities. The goal is to model traffic flow as a function of three variables, extract meaningful patterns through gradient analysis, and compute total flow capacity using numerical integration.

The Dataset

A 30-day traffic dataset was generated covering time of day, vehicle density, average speed, and traffic flow rate across Karachi, Lahore, and Islamabad. Flow is modeled using the fundamental relationship F = d x v with controlled noise added for realism. All data is saved to traffic_dataset.csv.

The Model

A linear regression model was fitted to the three-variable function F = f(t, d, v) using Scikit-learn. The resulting model is:

F = -2047.764 - 33.117t + 44.098d + 51.639v

Speed carries the largest positive weight at 51.639, followed by density at 44.098. Time has a negative coefficient of -33.117, meaning flow tends to decrease as the day progresses.

What the Code Does

Maximum and minimum flow identification: Peak flow hit 3,473 vehicles per hour at 1:00 PM with moderate density and solid speeds. Minimum flow dropped to 2,371 vehicles per hour during evening rush when density spiked to 94 vehicles per km and speed fell to 25 km per hour.

Gradient vector: Computed as (-33.117, 44.098, 51.639) at the reference point (10, 74, 40). This shows the direction of steepest ascent in the flow function — earlier time, higher density, higher speed.

Steepest descent: The negative gradient (-33.117 reversed, -44.098, -51.639) maps directly onto congestion onset conditions — later time, falling density efficiency, speed collapse.

Directional derivative: Computed for a congestion scenario where density increases while speed drops simultaneously. The result of -5.332 shows flow decreasing by roughly 5.33 vehicles per hour per unit movement in that direction — a gradual early-stage congestion signal.

Triple integral: Numerically integrated F over the region t in [6,10], d in [20,80], v in [30,80] using SciPy. Result: approximately 32,788,140 vehicles total accumulated flow across that operating range.

City comparison: Lahore recorded the highest average flow at 2,828.2 vehicles per hour. Karachi averaged 2,662.8 vehicles per hour with high density but low speed indicating genuine congestion. Islamabad logged the lowest at 2,590.8 vehicles per hour.

Visualization

Two scatter plots are generated showing flow versus density and flow versus speed across all 30 days. Saved as graphs.png.

Stack

Python, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib

To run:
python traffic_project.py
