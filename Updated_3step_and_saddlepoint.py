# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 21:12:43 2025

@author: Isuru Withanawasam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, epsilon_0
import mpmath as mp


# ==================== PARAMETERS ====================
I0_W_cm2 = 4e14          # laser intensity [W/cm²]
lambda0_nm = 800         # wavelength [nm]
Ip_eV = 21.56            # ionization potential [eV]

# Convert to SI
I0 = I0_W_cm2 * 1e4
lambda0 = lambda0_nm * 1e-9
Ip_J = Ip_eV * e

# Derived quantities
w0 = 2 * np.pi * c / lambda0
T0 = 2 * np.pi / w0
E0 = np.sqrt(2 * I0 / (c * epsilon_0))
Up_J = (e**2 * E0**2) / (4 * m_e * w0**2)
Up_eV = Up_J / e
E_w0 = 1240 / lambda0_nm  # photon energy [eV]


print("=" * 50)
print("SIMULATION PARAMETERS")
print("=" * 50)
print(f"Wavelength: {lambda0_nm} nm")
print(f"Intensity: {I0_W_cm2:.1e} W/cm²")
print(f"Up: {Up_eV:.2f} eV")
print(f"Cutoff: {Ip_eV + 3.17*Up_eV:.2f} eV")
print(f"Period T0: {T0*1e15:.2f} fs")
print("=" * 50 + "\n")

# ==================== CLASSICAL TRAJECTORIES ====================
def find_zero_crossing(x, y):
    """Find first upward zero-crossing with interpolation"""
    sign_y = np.sign(y)
    up_cross = np.where((sign_y[:-1] < 0) & (sign_y[1:] > 0))[0]
    
    if up_cross.size == 0:
        return 0
    
    ind = up_cross[0]
    x0 = x[ind] - y[ind] * (x[ind+1] - x[ind]) / (y[ind+1] - y[ind])
    return x0

def electron_trajectory(t, ti):
    """Electron position after ionization at ti"""
    if t < ti:
        return 0
    return (np.sin(w0*t) - np.sin(w0*ti) - (w0*t - w0*ti)*np.cos(w0*ti)) * (E0 * e / (m_e * w0**2))

# Time grids
t = np.linspace(0, 1.2*T0, 2000)
ti_grid = np.linspace(0, T0, 800)
trajectories = np.zeros((len(t), len(ti_grid)))

# Compute trajectories
t_return = np.zeros(len(ti_grid))
for i in range(len(ti_grid)):
    for j in range(len(t)):
        trajectories[j, i] = electron_trajectory(t[j], ti_grid[i])
    t_return[i] = find_zero_crossing(t, trajectories[:, i])

# Calculate kinetic and photon energies
ti_class, tr_class, E_photon = [], [], []

for i in range(len(ti_grid)):
    if t_return[i] > 0:
        ti_class.append(ti_grid[i])
        tr_class.append(t_return[i])
        v = (np.cos(w0*ti_grid[i]) - np.cos(w0*t_return[i])) * (E0 * e / (m_e * w0))
        Ek = 0.5 * m_e * v**2 / e
        E_photon.append(Ek + Ip_eV)

ti_class = np.array(ti_class)
tr_class = np.array(tr_class)
E_photon = np.array(E_photon)
ho_class = E_photon / E_w0  # harmonic order

# Separate short and long trajectories
tau = tr_class - ti_class
short_mask = tau < 0.65 * T0
long_mask = tau >= 0.65 * T0

ti_short = ti_class[short_mask]
tr_short = tr_class[short_mask]
ho_short = ho_class[short_mask]

ti_long = ti_class[long_mask]
tr_long = tr_class[long_mask]
ho_long = ho_class[long_mask]

print(f"Classical trajectories: {len(ti_class)} total")
print(f"  Short: {len(ti_short)}, Long: {len(ti_long)}\n")



# --- 1. Set Precision (Decimal Places) ---
mp.dps = 50

print(f"Using mpmath with {mp.dps} decimal places of precision.")

# --- 2. Define Physical Constants (in Atomic Units) ---
# [cite_start]Parameters based on Fig. 1 in the paper [cite: 130-131]

# --- Laser Parameters ---
LAMBDA_NM = mp.mpf('800.0')      # Wavelength (nm)
PULSE_T_FS = mp.mpf('8.0')        # Pulse duration (fs)
INTENSITY_WCM2 = mp.mpf('4e14') # Peak intensity (W/cm^2)
CEP = 0                           # Carrier-Envelope Phase (psi)

# --- Target Parameters (Argon) ---
IP_EV = mp.mpf('21.56')           # Ionization potential of Argon (eV)

# --- Conversion to Atomic Units (a.u.) ---
FS_TO_AU = mp.mpf('41.341')       # 1 fs = 41.341 a.u. of time
EV_TO_AU = mp.mpf(1.0 / 27.2114)  # 1 eV = 1/27.2114 a.u. (Hartree)
WCM2_TO_AU = mp.mpf(1.0 / 3.51e16) # 1 a.u. intensity = 3.51e16 W/cm^2

# --- Final Parameters in a.u. ---
Ip = IP_EV * EV_TO_AU                             # Ionization potential (a.u.)
w0 = mp.mpf('45.563') / LAMBDA_NM                  # Central frequency (a.u.)
T0 = 2.0 * mp.pi / w0                             # Optical cycle (a.u.)
I = INTENSITY_WCM2 * WCM2_TO_AU                   # Intensity (a.u.)
E0 = mp.sqrt(I)                                   # Peak electric field (a.u.)
T_pulse_au = PULSE_T_FS * FS_TO_AU                # Pulse duration (a.u.)

# tau from paper's pulse definition T = 2*tau*arccos(2^(-1/4))
tau = T_pulse_au / (2.0 * mp.acos(mp.power(2, -0.25)))

print(f"--- Atomic Units ---")
print(f"Ip (a.u.): {Ip}")
print(f"w0 (a.u.): {w0} (T0 = {T0} a.u.)")
print(f"E0 (a.u.): {E0}")
print(f"tau (a.u.): {tau}")
print("-" * 20)

# --- 2. Define Analytical Functions for Laser and Integrals ---
# Using mpmath functions (mp.sin, mp.cos, etc.)
w1 = w0
w2 = w0 + 2.0 / tau
w3 = w0 - 2.0 / tau

def A(t):
    #Analytical Vector Potential A(t) = -integral(E(t)) 
    term1 = mp.sin(w1 * t + CEP) / w1
    term2 = 0.5 * mp.sin(w2 * t + CEP) / w2
    term3 = 0.5 * mp.sin(w3 * t + CEP) / w3
    return -E0 / 2.0 * (term1 + term2 + term3)

def S_A(t):
    # Analytical Integral of A(t), S_A(t) = integral(A(t)) 
    term1 = -mp.cos(w1 * t + CEP) / (w1**2)
    term2 = -0.5 * mp.cos(w2 * t + CEP) / (w2**2)
    term3 = -0.5 * mp.cos(w3 * t + CEP) / (w3**2)
    return -E0 / 2.0 * (term1 + term2 + term3)




def p_s(ts, ts_prime):
    """ Calculates the stationary momentum p_s """
    if ts == ts_prime:
        return mp.mpc('1e100', '1e100') 
    integral_A = S_A(ts) - S_A(ts_prime)
    return (1.0 / (ts - ts_prime)) * integral_A

# --- 3. Define the System of Equations to Solve ---
def equations_to_solve(ts, ts_prime, omega):
    """
    Takes two complex mpmath numbers (ts, ts_prime) and the target omega
    Returns a tuple of two complex numbers (F1, F2) that should be zero.
    """
    try:
        ps_val = p_s(ts, ts_prime)
        A_ts = A(ts)
        A_ts_prime = A(ts_prime)
    except (ZeroDivisionError, OverflowError):
        return (mp.mpc('1e100', '1e100'), mp.mpc('1e100', '1e100'))

    # --- The Saddle-Point Equations ---
    F1 = 0.5 * (ps_val - A_ts_prime)**2 + Ip
    F2 = omega - 0.5 * (ps_val - A_ts)**2 - Ip
    return (F1, F2)

# --- 4. Loop Through Harmonics and Solve ---
harmonic_orders = range(13,75, 2)

# Lists to store the final results
short_results_t = []
short_results_t_prime = []
long_results_t = []
long_results_t_prime = []

for h_order in harmonic_orders:
    omega = h_order * w0
    
    # *** THIS LINE IS FIXED ***
    print(f"\n--- Solving for Harmonic {h_order} (w = {omega} a.u.) ---")

    # --- Define Guesses (these are generally robust enough for the whole range) ---
    ts_prime_re_guess_short = mp.mpf('25.0')
    ts_re_guess_short = mp.mpf('70.0')
    SHORT_path_guess = (
        mp.mpc(ts_re_guess_short, mp.mpf('1.5')),
        mp.mpc(ts_prime_re_guess_short, mp.mpf('1.5'))
    )
    
    ts_prime_re_guess_long = mp.mpf('20.0')
    ts_re_guess_long = mp.mpf('108.0')
    LONG_path_guess = (
        mp.mpc(ts_re_guess_long, mp.mpf('1.0')),
        mp.mpc(ts_prime_re_guess_long, mp.mpf('1.0'))
    )

    # --- Run Solver for SHORT Path ---
    try:
        ts_sol, ts_prime_sol = mp.findroot(
            lambda ts, ts_prime: equations_to_solve(ts, ts_prime, omega), 
            (SHORT_path_guess[0], SHORT_path_guess[1])
        )
        short_results_t.append(ts_sol)
        short_results_t_prime.append(ts_prime_sol)
        print(f"  SHORT path found.")
    except Exception:
        short_results_t.append(None)
        short_results_t_prime.append(None)
        print(f"  SHORT path FAILED.")

    # --- Run Solver for LONG Path ---
    try:
        ts_sol, ts_prime_sol = mp.findroot(
            lambda ts, ts_prime: equations_to_solve(ts, ts_prime, omega), 
            (LONG_path_guess[0], LONG_path_guess[1])
        )
        long_results_t.append(ts_sol)
        long_results_t_prime.append(ts_prime_sol)
        print(f"  LONG path found.")
    except Exception:
        long_results_t.append(None)
        long_results_t_prime.append(None)
        print(f"  LONG path FAILED.")

"""
# --- 5. Display Final Collected Data ---
print("\n" + "="*50)
print("           FINAL RESULTS")
print("="*50 + "\n")

print("--- Short Path Recombination Times (t_s) ---")
print(short_results_t)
print("\n--- Short Path Ionization Times (t_s') ---")
print(short_results_t_prime)
print("\n--- Long Path Recombination Times (t_s) ---")
print(long_results_t)
print("\n--- Long Path Ionization Times (t_s') ---")
print(long_results_t_prime)
"""

# --- Constants needed for plotting ---
FS_TO_AU = mp.mpf('41.341')

# --- 1. Extract Real Part of Times (in Femtoseconds) ---

# This helper function safely extracts the real part and converts to fs
def get_real_time_fs(complex_time_list):
    real_times_fs = []
    for t in complex_time_list:
        if t is not None:
            # Convert atomic units (a.u.) to femtoseconds (fs)
            real_times_fs.append(float(t.real / FS_TO_AU))
        else:
            real_times_fs.append(None) # Keep None for failed points
    return real_times_fs

# Create the lists for plotting
y_short_t = get_real_time_fs(short_results_t)
y_short_t_prime = get_real_time_fs(short_results_t_prime)
y_long_t = get_real_time_fs(long_results_t)
y_long_t_prime = get_real_time_fs(long_results_t_prime)

# --- 2. Create the Plot ---
plt.figure(figsize=(10, 6))


plt.plot(ti_short*1e15, ho_short, 'r-', markersize=3, label='Classical Short (ti)')
plt.plot(tr_short*1e15, ho_short, 'b-', markersize=3,  label='Classical Short (tr)')
plt.plot(ti_long*1e15, ho_long, color='green', markersize=3, label='Classical Long (ti)')
plt.plot(tr_long*1e15, ho_long, color='black', markersize=3,  label='Classical Long (tr)')


offset = 0.67
y_short_t_prime = [x + offset if x is not None else None for x in y_short_t_prime]
y_short_t = [x + offset if x is not None else None for x in y_short_t]
y_long_t_prime = [x + offset if x is not None else None for x in y_long_t_prime]
y_long_t = [x + offset if x is not None else None for x in y_long_t]


# Plot the SHORT paths
plt.plot(y_short_t_prime, harmonic_orders, 'r--', markersize=4, label='Saddle Short (ti)') # Red solid
plt.plot(y_short_t, harmonic_orders, 'b--', markersize=4, label='Saddle Short (tr)') # Blue solid

# Plot the LONG paths
plt.plot(y_long_t_prime, harmonic_orders, 'g--', markersize=5, label='Saddle Long (ti)') # Red dashed
plt.plot(y_long_t, harmonic_orders, 'k--', markersize=5, label='Saddle Long (ti)') # Blue dashed

# --- 3. Format the Plot ---
plt.ylabel('Harmonic Order')
plt.xlabel('Time (fs)')
#plt.title('Quantum Path Ionization and Recombination Times')
plt.legend(fontsize=12, ncol=2)
plt.grid(True)
plt.show()
