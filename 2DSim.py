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
    folder = r"E:\Data\2D_diffusion"        # Set folder and filename for saving data

    # Parameters (initialized locally)
    D_0 = 1.0               # Diffusion coefficient
    L = 10.0               # Domain from -L to L in both x and y
    N = 100                # Number of grid points in each direction
    dx = 2 * L / (N-1)     # Grid spacing
    x = np.linspace(-L, L, N)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    sig = 0.5
    m = 1.0              # Degree of nonlinearity
    t_end = 750.0          # Default end time
    scale_factor = 1.0

    # ----- Condition Selection -----
    def ask_conditions(current_m, current_D_0, current_t_end):
        root = tk.Tk()
        root.title("Select Conditions")
        
        window_width = 300
        window_height = 450 # Adjusted height for new field
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width - window_width) // 2
        pos_y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Default/initial values for StringVars
        initial_init_choice = "step"
        initial_bc_choice = "neumann (zero flux)"

        tk.Label(main_frame, text="Select initial condition type:").pack(pady=5)
        init_var = tk.StringVar(value=initial_init_choice)
        tk.OptionMenu(main_frame, init_var, "step", "gaussian").pack(pady=5)
        
        tk.Label(main_frame, text="Select boundary condition type:").pack(pady=5)
        bc_var = tk.StringVar(value=initial_bc_choice)
        tk.OptionMenu(main_frame, bc_var, "neumann (zero flux)", "dirichlet (zero density)").pack(pady=5)

        tk.Label(main_frame, text="Enter degree of nonlinearity (m):").pack(pady=5)
        m_entry = tk.Entry(main_frame)
        m_entry.insert(0, str(current_m))
        m_entry.pack(pady=5)

        tk.Label(main_frame, text="Enter diffusion coefficient (D_0):").pack(pady=5)
        D_0_entry = tk.Entry(main_frame)
        D_0_entry.insert(0, str(current_D_0))
        D_0_entry.pack(pady=5)

        tk.Label(main_frame, text="Enter end time (t_end):").pack(pady=5)
        t_end_entry = tk.Entry(main_frame)
        t_end_entry.insert(0, str(current_t_end))
        t_end_entry.pack(pady=5)
        
        # Store results here, initialized with defaults/passed-in parameters.
        # These will be updated if 'OK' is pressed and inputs are valid.
        dialog_results = {
            "init_choice": initial_init_choice,
            "bc_choice": initial_bc_choice,
            "m": current_m,
            "D_0": current_D_0,
            "t_end": current_t_end
        }

        def on_ok_pressed():
            # Update results from widgets before destroying the window
            dialog_results["init_choice"] = init_var.get()
            dialog_results["bc_choice"] = bc_var.get()
            try:
                dialog_results["m"] = float(m_entry.get())
            except ValueError:
                pass # Keep current_m if conversion fails (already in dialog_results)
            try:
                dialog_results["D_0"] = float(D_0_entry.get())
            except ValueError:
                pass # Keep current_D_0
            try:
                dialog_results["t_end"] = float(t_end_entry.get())
            except ValueError:
                pass # Keep current_t_end
            
            root.destroy()

        tk.Button(main_frame, text="OK", command=on_ok_pressed).pack(pady=10)
        
        root.mainloop() # This call blocks until root.destroy() is called
        
        # Return the stored/updated results
        return (
            dialog_results["init_choice"],
            dialog_results["bc_choice"],
            dialog_results["m"],
            dialog_results["D_0"],
            dialog_results["t_end"]
        )

    # Call ask_conditions with current local values and update them
    init_choice, bc_choice, m, D_0, t_end = ask_conditions(m, D_0, t_end)

    # Step function: density=1 inside a square |x|,|y| <=1 and 0 elsewhere.
    def initial_condition_step_2d(X_loc, Y_loc): # Renamed params
        n = np.zeros_like(X_loc)
        condition = (np.abs(X_loc) <= 1) & (np.abs(Y_loc) <= 1)
        n[condition] = 1.0
        return n

    # Gaussian initial condition (captures sig from main's scope)
    def initial_condition_gaussian_2d(X_loc, Y_loc): # Renamed params
        return np.exp(-((X_loc**2 + Y_loc**2) / (2 * sig**2)))
    
    if init_choice and init_choice.lower() == 'gaussian':
        n0 = initial_condition_gaussian_2d(X, Y)
    else:
        n0 = initial_condition_step_2d(X, Y)

    # Define non-linear diffusion term D_func (captures m, D_0 from main's scope)
    def D_func(u_val):
        return D_0 * (u_val ** m)

    # Flatten the 2D array since solve_ivp requires a 1D state vector. 
    n0_flat = n0.ravel()

    # Define diffusion_2d_func (captures N, dx, bc_choice, D_func from main's scope)
    def diffusion_2d_func(t_param, u_flat_param):
        u = u_flat_param.reshape((N, N))
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        
        dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        
        flux_x = D_func(u) * dudx
        flux_y = D_func(u) * dudy
        
        div_flux = np.zeros_like(u)
        div_flux[1:-1, :] += (flux_x[2:, :] - flux_x[:-2, :]) / (2 * dx)
        div_flux[:, 1:-1] += (flux_y[:, 2:] - flux_y[:, :-2]) / (2 * dx)

        if bc_choice and 'dirichlet' in bc_choice.lower():
            div_flux[0, :] = 0
            div_flux[-1, :] = 0
            div_flux[:, 0] = 0
            div_flux[:, -1] = 0
        if bc_choice and 'neumann' in bc_choice.lower(): # Corrected potential typo from 'nuemman' to 'neumann'
            div_flux[0, :] = div_flux[1, :]
            div_flux[-1, :] = div_flux[-2, :]
            div_flux[:, 0] = div_flux[:, 1]
            div_flux[:, -1] = div_flux[:, -2]

        return div_flux.ravel()

    # Set simulation time and time evaluation
    t_start = 0.0
    # t_end is now set from ask_conditions
    t_eval = np.linspace(t_start, t_end, 201)

    print("Solving PDE...")

    sol = solve_ivp(diffusion_2d_func, [t_start, t_end], n0_flat, method='BDF', t_eval=t_eval)

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

    for ti, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Processing front radii"):
        snapshot = sol.y[:, ti].reshape((N, N))

        interior_mask = np.ones_like(snapshot, dtype=bool)
        interior_mask[0, :] = interior_mask[-1, :] = interior_mask[:, 0] = interior_mask[:, -1] = False

        mask = (snapshot >= threshold) & interior_mask

        if mask.any():
            current_radius = np.max(np.sqrt(X[mask]**2 + Y[mask]**2))
        else:
            current_radius = front_radii[-1] if front_radii else 0.0

        if front_radii: # Ensure monotonic increase only if list is not empty
            current_radius = max(current_radius, front_radii[-1])

        front_radii.append(current_radius)

    margin = dx 
    cutoff = L - margin

    # Create a mask that excludes t=0 and times when the front is near the boundary
    # Ensure front_radii is a numpy array for boolean indexing with sol.t
    front_radii_np = np.array(front_radii)
    fit_mask = (sol.t > 0) & (front_radii_np < cutoff)
    
    # Ensure t_fit and front_fit are not empty before proceeding
    if np.any(fit_mask):
        t_fit = sol.t[fit_mask]
        front_fit = front_radii_np[fit_mask]

        # Take logarithms only if t_fit and front_fit are non-empty and positive
        if t_fit.size > 0 and front_fit.size > 0 and np.all(t_fit > 0) and np.all(front_fit > 0):
            log_t = np.log(t_fit)
            log_front = np.log(front_fit)

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
        else:
            print("Warning: Not enough valid data points for log-log fit of front radius.")
            a = np.nan # Set 'a' to nan if fit cannot be performed
            comparison = np.full_like(sol.t, np.nan) # Create NaN array for comparison
            log_t, log_front = np.array([]), np.array([]) # Empty arrays for plotting
    else:
        print("Warning: No data points satisfy the fitting criteria for front radius.")
        a = np.nan
        comparison = np.full_like(sol.t, np.nan)
        log_t, log_front = np.array([]), np.array([])


    # ------ Plotting -----

    # Plot front radius vs. time
    plt.figure()
    if not np.all(np.isnan(comparison)): # Plot fit only if 'a' was calculated
        plt.plot(sol.t, comparison, 'r-', label=f"t^a (a={a:.3f})" if not np.isnan(a) else "t^a (fit failed)")
    plt.plot(sol.t, front_radii_np, 'k--', label="Front Radius")
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
    root_msg = tk.Tk()
    root_msg.withdraw()

    save_data = messagebox.askyesno("Save Data", "Do you want to save the data from this trial?")
    if save_data:
        if not os.path.exists(folder):
            os.makedirs(folder)
        # date variable for folder name was defined earlier, use it or redefine
        trial_date = datetime.now().strftime("%Y%m%d_%H%M%S") 
        trial_folder = os.path.join(folder, f"trial_{t_end}_{trial_date}_m{m}_D{D_0}")
        os.makedirs(trial_folder)
        for i, t_val in tqdm(enumerate(sol.t), total=len(sol.t), desc="Saving timesteps"):
            if i == 0: # Exclude t=0 to save 200 files if 201 t_eval points
                continue
            snapshot = sol.y[:, i].reshape((N, N))
            df_snapshot = pd.DataFrame({ # Changed name to avoid conflict
                "X": np.repeat(x, N),
                "Y": np.tile(y, N),
                "U": snapshot.ravel()
            })
            filename = os.path.join(trial_folder, f"data_timestep_{i:03d}.csv")
            df_snapshot.to_csv(filename, index=False)
        print("Data saved to", trial_folder)
    else:
        print("Run data not saved.")

    root_msg.destroy() # Destroy the messagebox root

    # Save front data if fit was possible
    if np.any(fit_mask) and t_fit.size > 0 :
        print('Please select save option for front data...')
        root_front_msg = tk.Tk()
        root_front_msg.withdraw()
        save_front = messagebox.askyesno("Save Front Data", "Do you want to save the front data points used in the log plot?")
        if save_front:
            if not os.path.exists(folder):
                os.makedirs(folder)
            front_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            front_filename = os.path.join(folder, f"front_data_{front_date}.csv")
            front_df = pd.DataFrame({
                "Time": t_fit,
                "FrontRadius": front_fit,
                "logTime": log_t, # Use already computed log_t
                "logFront": log_front # Use already computed log_front
            })
            front_df.to_csv(front_filename, index=False)
            print("Front data saved to", front_filename)
        else:
            print("Front data not saved.")
        root_front_msg.destroy()
    else:
        print("Front data not saved as fitting was not performed or yielded no data.")

if __name__ == "__main__":
    main()
