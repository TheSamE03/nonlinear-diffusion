import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

def main():
    # Parameters
    Deff = 1
    L = 10.0
    N = 400     # Number of grid points
    dx = L / (N-1)
    x = np.linspace(-L, L, N)
    sig = 0.5
    m = 1.0       # Degree of nonlinearity
    a = 1.0/(2.0 + m)     # Where m is the degree of nonlinear term n(x, t)
    scale_factor = 1.0

    # define non-linear diffusion term: 
    def D(u):
        return u**m

    # Step function initial condition
    def initial_condition_step(x):
        n = np.zeros_like(x)
        n[np.where((x >= -1) & (x <= 1))] = 1.0
        return n
    
    # Gaussian initial condition
    def initial_condition_gaussian(x):
        return np.exp(-((x**2)/((2*sig)**2)))
    
    # Smooth step function using tanh (epsilon controls the width of the transition)
    def initial_condition_sm_step(x, eps=0.05):
        return 0.5 * (1 + np.tanh((x + 1) / eps)) - 0.5 * (1 + np.tanh((x - 1) / eps))
    
    # Offset initial condition
    def initial_condition_offset(x, shift):
        n = np.zeros_like(x)
        condition = (np.abs(x - shift) < 1)
        n[condition] = 1.0
        return n
    
    def initial_condition_amplified(x, amplitude):
        n = np.zeros_like(x)
        n[np.where((x >= -1) & (x <= 1))] = amplitude
        return n

    # Choose initial condition

    n0 = initial_condition_step(x)
    # n0 = initial_condition_gaussian(x)
    # n0 = initial_condition_sm_step(x, eps=0.1)
    # n0 = initial_condition_offset(x, -5.0)
    # n0 = initial_condition_offset(x, -9.0)
    # n0 = initial_condition_amplified(x, 0.5)

    # Time loop (test with linear diffusion (D = 1))
    def dn_dt(t, n):
        global dndt
        flux = Deff * 0.5 *(n[:-1] + n[1:]) * ((n[1:] - n[:-1]) / dx)
        dndt = np.zeros_like(n)
        dndt[1:-1] = (flux[1:] - flux[:-1]) / dx
        dndt[0] = 0.0
        dndt[-1] = 0.0
        return dndt
    
    # Time loop for non-linear diffusion 
    def dn_dt_nonlinear(t, n):
        global dndt
        flux = 0.5 * (D(n[:-1]) + D(n[1:])) * ((n[1:] - n[:-1]) / dx)
        dndt = np.zeros_like(n)
        dndt[1:-1] = (flux[1:] - flux[:-1]) / dx
        dndt[0] = 0.0
        dndt[-1] = 0.0
        return dndt
    
    def dn_dt_nl_insulated(t, n):
        global dndt
        flux = 0.5 * (D(n[:-1]) + D(n[1:])) * ((n[1:] - n[:-1]) / dx)
        # Insert BCs
        flux_con = np.concatenate(([0], flux, [0]))
        dndt = (flux_con[1:] - flux_con[:-1]) / dx
        return dndt

    t_start = 0.0
    t_end = 100.0
    t_tot = np.linspace(t_start, t_end, 1001)
    
    # This is for visualizing the early time behavior
    t_dense = np.linspace(t_start, 0.05, 51)      # Dense sampling for 0 <= t <= 1
    t_sparse = np.linspace(0.05, t_end, 50)         # Coarser sampling for 1 < t <= 50
    # t_tot = np.unique(np.concatenate((t_dense, t_sparse)))
    

    sol = solve_ivp(dn_dt_nl_insulated, [t_start, t_end], n0, method = 'BDF', t_eval=t_tot)

    data = np.column_stack((sol.t, sol.y.T))
    folder = r"C:\Users\erneb\Documents\School\Research\Non-linear Spreading\data"

    date = datetime.now().strftime("%Y%m%d%H%M%S")

    root = tk.Tk()
    root.withdraw()
    save_data = messagebox.askyesno("Save Run Data", "Do you want to save this run's data?")

    if save_data:
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"dataset_{date}.csv")
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        print("Data saved to", filename)
    else:
        print("Run data not saved.")
    root.destroy()
    
    # Reevaluate this and why it doesnt stop at the boundary
    def front_position(n, x, threshold=0.01):
        indices = np.where(n >= threshold)[0]

        if indices.size == 0:
            return np.nan  # No point meets the threshold
        idx = indices[-1]

        if idx == len(x)-1:
            return x[idx]
        
        n1, n2 = n[idx], n[idx+1]
        x1, x2 = x[idx], x[idx+1]

        # Avoid division by zero
        if n1 == n2:
            return x1
        frac = (threshold - n1) / (n2 - n1)
        return x1 + frac * (x2 - x1)
    
    X_front = [front_position(sol.y[:, i], x, threshold=0.01) for i in range(sol.y.shape[1])]

    plt.plot(t_tot, X_front, 'k--', label="Front Position")
    plt.xlabel("Time")
    plt.ylabel("Front Position")
    plt.title("Front Position vs. Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the solution
    for i, t in enumerate(sol.t):
    # Plot every 10th time step for clarity
        if i % 100.0 == 0:
            plt.plot(x, sol.y[:, i], label=f"t = {t:.2f}")
        
    n_final = sol.y[:, -1]
    total = np.trapezoid(n_final, x)
    mean = np.trapezoid(x * n_final, x) / total
    variance = np.trapezoid((x - mean)**2 * n_final, x) / total

    plt.xlabel("x")
    plt.xlim(-12, 12)
    plt.ylim(0, 1.2)
    plt.ylabel("Electron density n(x,t)")
    plt.title("Diffusion of Electron Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
