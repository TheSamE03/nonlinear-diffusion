import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tqdm import tqdm 
import tkinter.simpledialog as simpledialog

'''
This is a 2D siumulation for nonlinear charge ensity diffusion.
The diffusion term is defined as D(u) = u^m, where m is the degree of nonlinearity.

The user should specify the folder variable as the location to save data.

For better performace on devices with limited  memory/CPU power, set N to ~50.
This is the "resolution" of the simulation. Lower N values will be fewer grid points for 
the simulation to compute at each timestep. 

By Samuel Erne, 2025
'''

def main():
    folder = r"D:\Data\2D_diffusion"        # Set folder and filename for saving data

    # Parameters
    D_0 = 1.0               # Diffusion coefficient
    L = 10.0               # Domain from -L to L in both x and y
    N = 50                # Number of grid points in each direction
    dx = 2 * L / (N-1)     # Grid spacing
    x = np.linspace(-L, L, N)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    sig = 0.5
    m = 1.0              # Degree of nonlinearity
    scale_factor = 1.0

    # ----- Condition Selection -----
    def ask_conditions():
        root = tk.Tk()
        root.title("Select Conditions")
        
        # Initial condition
        tk.Label(root, text="Select initial condition type:").pack(pady=5)
        init_var = tk.StringVar(value="step")
        tk.OptionMenu(root, init_var, "step", "gaussian").pack(pady=5)
        
        # Boundary condition
        tk.Label(root, text="Select boundary condition type:").pack(pady=5)
        bc_var = tk.StringVar(value="neumann")
        tk.OptionMenu(root, bc_var, "neumann", "dirichlet").pack(pady=5)

        tk.Label(root, text="Enter degree of nonlinearity:").pack(pady=5)
        m = tk.Entry(root)
        m.insert(0, "1.0")
        m.pack(pady=5)

        tk.Label(root, text="Enter diffusion coefficient:").pack(pady=5)
        D_0 = tk.Entry(root)
        D_0.insert(0, "1.0")
        D_0.pack(pady=5)
        
        tk.Button(root, text="OK", command=root.destroy).pack(pady=10)
        root.mainloop()
        
        return init_var.get(), bc_var.get()

    init_choice, bc_choice = ask_conditions()

    # Step function: density=1 inside a square |x|,|y| <=1 and 0 elsewhere.
    def initial_condition_step_2d(X, Y):
        n = np.zeros_like(X)
        condition = (np.abs(X) <= 1) & (np.abs(Y) <= 1)
        n[condition] = 1.0
        return n

    # Gaussian initial condition
    def initial_condition_gaussian_2d(X, Y):
        return np.exp(-((X**2 + Y**2) / (2 * sig**2)))
    
    if init_choice and init_choice.lower() == 'gaussian':
        n0 = initial_condition_gaussian_2d(X, Y)
    else:
        n0 = initial_condition_step_2d(X, Y)

    # Define non-linear diffusion term: D(u) = u^m
    def D(u):
        return D_0*u**m

    # Flatten the 2D array since solve_ivp requires a 1D state vector. 
    n0_flat = n0.ravel()

    def diffusion_2d(t, u_flat):
        # Reshape the 1D state to a 2D grid
        u = u_flat.reshape((N, N))
        # Compute gradients with central differences
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        # Interior points (central differences)
        dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        # Compute flux
        flux_x = D(u) * dudx
        flux_y = D(u) * dudy
        # Compute divergence of the flux
        div_flux = np.zeros_like(u)
        div_flux[1:-1, :] += (flux_x[2:, :] - flux_x[:-2, :]) / (2 * dx)
        div_flux[:, 1:-1] += (flux_y[:, 2:] - flux_y[:, :-2]) / (2 * dx)

        if bc_choice and bc_choice.lower() == 'dirichlet':
            # For Dirichlet (0 density), force the time derivative to zero at the boundaries.
            div_flux[0, :] = 0
            div_flux[-1, :] = 0
            div_flux[:, 0] = 0
            div_flux[:, -1] = 0
        if bc_choice and bc_choice.lower() == 'nuemman':
            # For Neumann (0 flux), use central differences to mimic zero derivative:
            div_flux[0, :] = div_flux[1, :]
            div_flux[-1, :] = div_flux[-2, :]
            div_flux[:, 0] = div_flux[:, 1]
            div_flux[:, -1] = div_flux[:, -2]

        # Return the time derivative (flattened)
        return div_flux.ravel()

    # Set simulation time and time evaluation
    t_start = 0.0
    t_end = 750.0   
    t_eval = np.linspace(t_start, t_end, 201)

    print("Solving PDE...")

    sol = solve_ivp(diffusion_2d, [t_start, t_end], n0_flat, method='BDF', t_eval=t_eval)

    print("Processing results...")

    # ----- Save Data -----
    # Combine time and solution data. Note: sol.y shape is (N*N, n_t)
    data = np.column_stack((sol.t, sol.y.T))
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    header = "Time," + ",".join([f"u_{i}" for i in range(N*N)])

    print("Formatting data...")

    rows = []
    for ti, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Processing time steps"):
        snapshot = sol.y[:, ti].reshape((N, N))
        for i in range(N):
            for k in range(N):
                rows.append({"Time": t_val, "X": x[i], "Y": y[k], "U": snapshot[i, k]})
                
    df = pd.DataFrame(rows)

    # ----- Front Tracking -----

    threshold = 0.01        # Threshold for defining the front
    front_radii = []

    '''
    for ti, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Tracking Fronts"):
        snapshot = sol.y[:, ti].reshape((N, N))
        R = np.sqrt(X**2 + Y**2)
        mask = snapshot >= threshold
        front_radius = np.max(R[mask]) if mask.any() else 0.0
        front_radii.append(front_radius)
    '''

    for ti, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Processing front radii"):
        snapshot = sol.y[:, ti].reshape((N, N))

        interior_mask = np.ones_like(snapshot, dtype=bool)
        interior_mask[0, :] = interior_mask[-1, :] = interior_mask[:, 0] = interior_mask[:, -1] = False

        mask = (snapshot >= threshold) & interior_mask

        if mask.any():
            current_radius = np.max(np.sqrt(X[mask]**2 + Y[mask]**2))
        else:
            current_radius = front_radii[-1] if front_radii else 0.0

        # Enforce monotonic increase; necessary for 0 flux boundary conditions
        if front_radii:
            current_radius = max(current_radius, front_radii[-1])

        front_radii.append(current_radius)

    margin = dx 
    cutoff = L - margin

    # Create a mask that excludes t=0 and times when the front is near the boundary
    fit_mask = (sol.t > 0) & (np.array(front_radii) < cutoff)

    t_fit = sol.t[fit_mask]
    front_fit = np.array(front_radii)[fit_mask]


    # Take logarithms
    log_t = np.log(t_fit)
    log_front = np.log(front_fit)

    # Fit a line and extract the slope which represents the exponent a
    coeffs = np.polyfit(log_t, log_front, 1)
    a = coeffs[0]
    intercept = coeffs[1]
    print(f"Fitted exponent a: {a:.3f}")

    plt.figure()
    plt.plot(log_t, log_front, 'ko', label="Data (log-scale)")
    plt.plot(log_t, a * log_t + intercept, 'r-', label=f"Fit (a = {a:.3f})")
    plt.xlabel("log(Time)")
    plt.ylabel("log(Front Radius)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    comparison = np.exp(intercept)*np.power(sol.t, a)

    # ------ Plotting -----

    # Plot front radius vs. time
    plt.figure()
    plt.plot(sol.t, comparison, 'r-', label="t^a")
    plt.plot(sol.t, front_radii, 'k--', label="Front Radius")
    plt.xlabel("Time")
    plt.ylabel("Front Radius")
    plt.title("Front Radius vs. Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the final state of the density as a contour plot
    final_state = sol.y[:, -1].reshape((N, N))
    plt.figure()
    plt.contourf(X, Y, final_state, levels=50, cmap='viridis')
    plt.colorbar(label="Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"State after {t_end} time units")
    plt.tight_layout()
    plt.show()

    print('Please select save option...')
    root = tk.Tk()
    root.withdraw()

    save_data = messagebox.askyesno("Save Data", "Do you want to save the data from this trial?")
    if save_data:
        if not os.path.exists(folder):
            os.makedirs(folder)
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_folder = os.path.join(folder, f"trial_{t_end}_{date}")
        os.makedirs(trial_folder)
        # Exclude the first timestep (t=0) so that we save 200 files.
        for i, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Saving timesteps"):
            if i == 0:
                continue
            snapshot = sol.y[:, i].reshape((N, N))
            # Create a DataFrame with all combinations of x,y and corresponding u value.
            df = pd.DataFrame({
                "X": np.repeat(x, N),
                "Y": np.tile(y, N),
                "U": snapshot.ravel()
            })
            filename = os.path.join(trial_folder, f"data_timestep_{i:03d}.csv")
            df.to_csv(filename, index=False)
        print("Data saved to", trial_folder)
    else:
        print("Run data not saved.")

    root.destroy()

    print('Please select save option for front data...')
    root = tk.Tk()
    root.withdraw()

    save_front = messagebox.askyesno("Save Front Data", "Do you want to save the front data points used in the log plot?")
    if save_front:
        if not os.path.exists(folder):
            os.makedirs(folder)
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        front_filename = os.path.join(folder, f"front_data_{date}.csv")
        # Create a DataFrame with time, front radius, and their logarithms
        front_df = pd.DataFrame({
            "Time": t_fit,
            "FrontRadius": front_fit,
            "logTime": np.log(t_fit),
            "logFront": np.log(front_fit)
        })
        front_df.to_csv(front_filename, index=False)
        print("Front data saved to", front_filename)
    else:
        print("Front data not saved.")

    root.destroy()

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

    # Subplot 1: Log-Log Plot of Front Radius vs Time with fit line
    axs[0].plot(log_t, log_front, 'ko', label="Data (log-scale)")
    axs[0].plot(log_t, a * log_t + intercept, 'r-', label=f"Fit (a = {a:.3f})")
    axs[0].set_xlabel("log(Time)")
    axs[0].set_ylabel("log(Front Radius)")
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].legend()
    axs[0].set_title("Log-Log Plot of Front Radius vs Time")

    # Subplot 2: Front Radius vs Time Plot
    axs[1].plot(sol.t, comparison, 'r-', label="t^a")
    axs[1].plot(sol.t, front_radii, 'k--', label="Front Radius")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Front Radius")
    axs[1].set_title("Front Radius vs Time")
    axs[1].set_aspect('equal', adjustable='box')

    axs[1].legend()

    # Subplot 3: Final State Contour Plot
    c = axs[2].contourf(X, Y, final_state, levels=50, cmap='viridis')
    fig.colorbar(c, ax=axs[2], label="Density")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title(f"State after {t_end} Time Units")
    axs[2].set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()
