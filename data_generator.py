import numpy as np

def get_random_start():
    """
    This function uses random keplarian parameters to generate the random starting
    point for our Runka Kutta integrator to simulate the orbit.
    """
    G = 6.6743e-11  # Gravitational constant in m^3 kg^-1 s^-2

    # Randomly sample the parameters
    a = np.random.uniform(0.5, 2.0) * 1.496e11 #convert to meters # Semi-major axis in AU
    e = np.random.uniform(0.0, 0.9)   # Eccentricity
    omega = np.random.uniform(0, 2 * np.pi)  # Argument of periapsis in radians
    M0 = np.random.uniform(0, 2 * np.pi)     # Mean anomaly at epoch in radians
    mass = np.random.uniform(1e24, 1e30)     # Mass in kg

    n = np.sqrt(G * mass / a**3)
    E = M0  # Good initial guess
    for _ in range(100):  # 10 to 20 iterations is usually plenty
        E_next = E - (E - e * np.sin(E) - M0) / (1.0 - e * np.cos(E))
        if abs(E_next - E) < 1e-6:
            E = E_next
            break
        E = E_next
    #print(E)
    x1 = a * (np.cos(E) - e)
    y1 = a * np.sqrt(1 - e**2) * np.sin(E)
    vx1 = -n * a / (1 - e * np.cos(E)) * np.sin(E)
    vy1 = n * a / (1 - e * np.cos(E)) * np.sqrt(1 - e**2) * np.cos(E)

    x0 = (x1*np.cos(omega) - y1*np.sin(omega))
    y0 = (x1*np.sin(omega) + y1*np.cos(omega))
    vx0 = (vx1*np.cos(omega) - vy1*np.sin(omega))
    vy0 = (vx1*np.sin(omega) + vy1*np.cos(omega))

    params = [a, e, omega, M0, mass]
    return np.array([x0, y0, vx0, vy0]), params

def Comet_system(state, M):
    """
    This function returns the dx, dy, dz based on the current state
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    G = 6.6743e-11
    # The state vector is of the form [x, y, vx, vy]
    return np.array([vx, vy, -G*M*x/(r**3), -G*M*y/(r**3)])

def RK4_anyD(fun, y0, M, t_final, target_error):
    # I had to remove preallocation due to dynamic step size
    states = [np.array(y0)]
    times = [0.0]
    dt = 1000
    
    while times[-1] < t_final:
        # Take a full step
        y = states[-1]
        k1 = fun(y, M)*dt
        k2 = fun(y + k1 /2, M)*dt
        k3 = fun(y + k2 /2, M)*dt
        k4 = fun(y + k3, M)*dt

        full_step = y + (k1 + 2*k2 + 2*k3 + k4) * (1 / 6)
        
        #take 2 half steps
        y = states[-1]
        for i in [1,2]:
            k1 = fun(y, M)*dt/2
            k2 = fun(y + k1 /2, M)*dt/2
            k3 = fun(y + k2 /2, M)*dt/2
            k4 = fun(y + k3, M)*dt/2
            y += (k1 + 2*k2 + 2*k3 + k4) * (1 / 6)

        two_steps = y

        e = np.linalg.norm(full_step[:2]-two_steps[:2])
        
        if e == 0:
            e = 1e-10

        error_rate = e/dt
        
        if error_rate < target_error: #one km / year is about 3.17e-5 m/s
            states.append(full_step + (1/15)*(full_step-two_steps))
            times.append(times[-1] + dt)
            #print(dt)
        
        scaling_factor = (target_error / error_rate)
        dt = dt * min(2.0, max(0.1, 0.9 * scaling_factor))
        
    return np.array(states), np.array(times)

def get_seconds(years):
    return years * 365.25 * 24 * 3600

filename = "orbits_data.h5"
import h5py

for data_point in range(10000):
    initial_state, params = get_random_start()
    S, T = RK4_anyD(Comet_system, initial_state, params[4], get_seconds(3), 1e-5)

    index = np.random.randint(0, len(S), 64)
    index.sort()
    points = np.column_stack((S[index, 0], S[index, 1], T[index]))

    #inject noise up to 1M km in x and y, but not in time
    points_noisy = np.column_stack((points[:, 0] + np.random.normal(0, 1e9, size=points.shape[0]),
                                    points[:, 1] + np.random.normal(0, 1e9, size=points.shape[0]),
                                    points[:, 2]))

    with h5py.File(filename, "a") as f:
        group_name = f"orbit_{data_point}"
        group = f.create_group(group_name)
        group.create_dataset("points", data=points)
        group.create_dataset("points_noisy", data=points_noisy)

        group.attrs["mass"] = params[4]
        group.attrs["a"] = params[0]
        group.attrs["e"] = params[1]
        group.attrs["omega"] = params[2]
        group.attrs["M0"] = params[3]
    
    if data_point % 50 == 0:
        print(f"Saved {data_point} orbits to {filename}")
