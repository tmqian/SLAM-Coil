from field import *
from racetrack import *
from scipy.signal import find_peaks
from scipy.optimize import least_squares

def Optimize(plot: bool, rt: Racetrack, coil_ref: str, coil_idx: int = 0, target_B: float = 0.25, current_per_type: bool = False):
    axis_path = rt.axis_path
    B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path) 

    
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

    first_quad_coils = [c for c in rt.coils if c.type in rt.center_types and c.Xc > 0 and c.Yc >= 0]

    # Preserve deterministic order from the actual coil list.
    ordered_types = []
    for coil in rt.coils:
        if coil.type in rt.center_types and coil.type not in ordered_types:
            ordered_types.append(coil.type)

    if current_per_type:
        opt_labels = ordered_types
        initial_guess = np.array([
            next(c.current for c in rt.coils if c.type == typ)
            for typ in opt_labels
        ], dtype=float)
    else:
        opt_labels = first_quad_coils
        initial_guess = np.array([coil.current for coil in first_quad_coils], dtype=float)

    lower_bounds = np.zeros(len(opt_labels), dtype=float)
    upper_bounds = np.full(len(opt_labels), 1500.0, dtype=float)

    def apply_currents(currents):
        if current_per_type:
            for typ, I in zip(opt_labels, currents):
                for coil in rt.coils:
                    if coil.type == typ:
                        coil.current = float(I)
        else:
            for coil, I in zip(opt_labels, currents):
                group = get_symmetric_group(coil, rt.coils)
                for symmetric_coil in group:
                    symmetric_coil.current = float(I)

    def B_field(s, currents):
        apply_currents(currents)

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

    #----------------"real time" plotting----------------
    it_counter = 0
    def plot_callback(xk):
        nonlocal it_counter
        it_counter += 1

        B_mag, _ = get_Bmag_on_axis(rt.coils, axis_path)
        itteration_line.set_ydata(B_mag)
        ax.set_title(f"Iteration {it_counter}")
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

    else:   
        result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), args=(target_B,))

    # Ensure the coil objects hold the optimized values after solver termination.
    apply_currents(result.x)

    if current_per_type:
        optimization = {typ: float(I) for typ, I in zip(opt_labels, result.x)}
    else:
        optimization = {
            f"{coil.type}[{idx}]": float(I)
            for idx, (coil, I) in enumerate(zip(opt_labels, result.x))
        }

    print(f"Optimized currents: {optimization}")
    return result