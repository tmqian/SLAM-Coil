from field import *
from racetrack import *
from scipy.signal import find_peaks
from scipy.optimize import least_squares

def Optimize(plot: bool, rt: Racetrack, coil_ref: str, coil_idx: int = 0, target_B: float = 0.25):
    axis_path = rt.axis_path
    B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path)
    _, idx_ends = get_coil_scoord(rt.coils, axis_path, coil_ref)

    
    def get_symmetric_group(coil, all_coils):
        """Find all symmetric partners of a coil"""
        group = {coil}
        tolerance = 1e-6
        for other in all_coils:
            if (abs(other.Xc - (-coil.Xc)) < tolerance and abs(other.Yc - coil.Yc) < tolerance) or \
            (abs(other.Xc - coil.Xc) < tolerance and abs(other.Yc - (-coil.Yc)) < tolerance) or \
            (abs(other.Xc - (-coil.Xc)) < tolerance and abs(other.Yc - (-coil.Yc)) < tolerance):
                group.add(other)
        return group

    def B_field(s, currents):
        first_quad_coils = [c for c in rt.coils if c.type in rt.center_types and c.Xc > 0 and c.Yc >= 0]
        
        for (coil, I) in zip(first_quad_coils, currents):
            group = get_symmetric_group(coil, rt.coils)
            for symmetric_coil in group:
                symmetric_coil.current = I

        B_mag, _ = get_Bmag_on_axis(rt.coils, axis_path)

        _, idx_ends = get_coil_scoord(rt.coils, axis_path, coil_ref)

        #find index for middle of stellarator section
        target_point = np.array([rt.Mirror_Length + rt.Stellerator_Radius, 0])
        distances = np.linalg.norm(axis_path - target_point, axis=1)
        idx_center = np.argmin(distances)

        B_region = B_mag[idx_ends[coil_idx]:idx_center]
        return B_region


    def residuals(currents, target_value):
        B_region = B_field(None, currents)
        target_B = np.full_like(B_region, target_value)
        return B_region - target_B

    lower_bounds = np.zeros(len([coil for coil in rt.coils if coil.type in rt.center_types and coil.Xc > 0 and coil.Yc >= 0]))
    upper_bounds = np.full(len([coil for coil in rt.coils if coil.type in rt.center_types and coil.Xc > 0 and coil.Yc >= 0]), 1500.0)
    initial_guess = np.array([coil.current for coil in rt.coils if coil.type in rt.center_types and coil.Xc > 0 and coil.Yc >= 0])

    #----------------"real time" plotting----------------
    it_counter = 0
    def plot_callback(xk):
        nonlocal it_counter
        it_counter += 1

        B_mag, _ = get_Bmag_on_axis(rt.coils, axis_path)
        itteration_line.set_ydata(B_mag)
        peak_idx, _ = find_peaks(B_mag[idx_ends[0]:idx_ends[1]])
        trough_idx, _ = find_peaks(-B_mag[idx_ends[0]:idx_ends[1]])
        if len(peak_idx) > 0 and len(trough_idx) > 0:
            ripple = (np.max(B_mag[idx_ends[0]:idx_ends[1]][peak_idx]) - np.min(B_mag[idx_ends[0]:idx_ends[1]][trough_idx])) / np.max(B_mag[idx_ends[0]:idx_ends[1]])
        else:
            ripple = 0.0
        ax.set_title(f"Iteration {it_counter}, Ripple: {ripple:.4%}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))


        itteration_line, = ax.plot(s_coord, np.zeros_like(s_coord), color='red', lw=2, label='Current Fit')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 0.5)
        fig.canvas.draw()
        ax.plot(s_coord, B_mag, color='blue', lw=2, label='B-field Region', linestyle='--')

        result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), args=(target_B,), callback=plot_callback)

        plt.ioff()
        plt.legend()
        plt.show()

        optimized_currents = result.x
        optimization = {coil.type: float(I) for coil, I in zip([coil for coil in rt.coils if coil.type in rt.center_types and coil.Xc > 0 and coil.Yc >= 0], optimized_currents)}
        print(f"Optimized currents: {optimization}")

    else:   
        result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), args=(target_B,))