import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# 1. Data reading
# ====================================
file_path = '/root/autodl-tmp/raw.xlsx'
data = pd.read_excel(file_path)
trajectory = data.iloc[:, 0:3].values

time = trajectory[:, 0]
azimuth_raw = trajectory[:, 1].copy()
polar_raw = trajectory[:, 2].copy()


# ====================================
# 2. Sliding window neighbor-frame angle correction algorithm
# ====================================
def correct_angles_neighbor_window(azimuth_raw, polar_raw, window_size=10):
    N = len(azimuth_raw)
    azimuth_corr = np.zeros(N)
    polar_corr = np.zeros(N)

    azimuth_corr[0] = azimuth_raw[0]
    polar_corr[0] = polar_raw[0]

    k_candidates = [-2, -1, 0, 1, 2]

    for i in range(1, N):
        start_idx = max(0, i - window_size)
        local_mean_az = np.mean(azimuth_corr[start_idx:i])

        best_k = 0
        best_diff = float('inf')

        for k in k_candidates:
            cand_az = azimuth_raw[i] + 180 * k
            if k % 2 != 0:
                cand_polar = 180 - polar_raw[i]
            else:
                cand_polar = polar_raw[i]

            diff = abs(cand_az - local_mean_az)

            if diff < best_diff:
                best_diff = diff
                best_k = k

        azimuth_corr[i] = azimuth_raw[i] + 180 * best_k
        if best_k % 2 != 0:
            polar_corr[i] = 180 - polar_raw[i]
        else:
            polar_corr[i] = polar_raw[i]

    return azimuth_corr, polar_corr


# ====================================
# 3. Apply correction
# ====================================
window_size = 10
azimuth_corrected, polar_corrected = correct_angles_neighbor_window(
    azimuth_raw, polar_raw,
    window_size=window_size
)


# ====================================
# 4. Compute local mean line (for visualization)
# ====================================
def compute_local_mean(azimuth_corr, window_size):
    """Compute the local mean of the previous window_size frames at each time step (for display)."""
    N = len(azimuth_corr)
    local_mean = np.zeros(N)
    local_mean[0] = azimuth_corr[0]

    for i in range(1, N):
        start_idx = max(0, i - window_size)
        local_mean[i] = np.mean(azimuth_corr[start_idx:i])

    return local_mean


local_mean_az = compute_local_mean(azimuth_corrected, window_size)

# ====================================
# 5. Visualization (with local mean line, no difference plot)
# ====================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ---- Azimuth raw vs corrected + local mean line ----
axes[0, 0].plot(time, azimuth_raw, 'r--', alpha=0.5, label='Raw')
axes[0, 0].plot(time, azimuth_corrected, 'b-', linewidth=1.5, label='Corrected')
axes[0, 0].plot(time, local_mean_az, 'g:', linewidth=1.5, alpha=0.8, label=f'Local Mean (w={window_size})')
axes[0, 0].set_title('Azimuth Angle')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Azimuth (°)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ---- Polar raw vs corrected ----
axes[0, 1].plot(time, polar_raw, 'r--', alpha=0.5, label='Raw')
axes[0, 1].plot(time, polar_corrected, 'b-', linewidth=1.5, label='Corrected')
axes[0, 1].set_title('Polar Angle')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Polar (°)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# ---- Azimuth candidate point illustration (show candidates and selected for each raw frame) ----
# Select some frames to show candidate points
sample_indices = np.linspace(0, len(time) - 1, min(20, len(time)), dtype=int)
axes[1, 0].plot(time, azimuth_raw, 'r--', alpha=0.3, label='Raw')
axes[1, 0].plot(time, azimuth_corrected, 'b-', linewidth=1.5, label='Corrected')

for idx in sample_indices:
    if idx == 0:
        continue
    # Show the candidate points for this frame
    for k in [-2, -1, 0, 1, 2]:
        cand = azimuth_raw[idx] + 180 * k
        color = 'g' if cand == azimuth_corrected[idx] else 'gray'
        alpha = 0.8 if cand == azimuth_corrected[idx] else 0.3
        marker = 'o' if cand == azimuth_corrected[idx] else 'x'
        size = 40 if cand == azimuth_corrected[idx] else 20
        axes[1, 0].scatter(time[idx], cand, c=color, marker=marker, s=size, alpha=alpha, zorder=5)

axes[1, 0].set_title('')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Azimuth (°)')
axes[1, 0].legend(['Raw', 'Corrected', 'Selected', 'Rejected'])
axes[1, 0].grid(True, alpha=0.3)

# ---- Polar candidate point illustration ----
axes[1, 1].plot(time, polar_raw, 'r--', alpha=0.3, label='Raw')
axes[1, 1].plot(time, polar_corrected, 'b-', linewidth=1.5, label='Corrected')

for idx in sample_indices:
    if idx == 0:
        continue
    for k in [-2, -1, 0, 1, 2]:
        if k % 2 != 0:
            cand = 180 - polar_raw[idx]
        else:
            cand = polar_raw[idx]
        color = 'g' if cand == polar_corrected[idx] else 'gray'
        alpha = 0.8 if cand == polar_corrected[idx] else 0.3
        marker = 'o' if cand == polar_corrected[idx] else 'x'
        size = 40 if cand == polar_corrected[idx] else 20
        axes[1, 1].scatter(time[idx], cand, c=color, marker=marker, s=size, alpha=alpha, zorder=5)

axes[1, 1].set_title('')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Polar (°)')
axes[1, 1].legend(['Raw', 'Corrected', 'Selected', 'Rejected'])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'angle_correction_neighbor_window{window_size}.png', dpi=150)
plt.show()

# ====================================
# 6. Output statistics
# ====================================
print("\n" + "=" * 50)
print(f"Sliding window neighbor-frame angle correction statistics (window_size={window_size})")
print("=" * 50)
print(f"Total frames: {len(time)}")
print(f"Azimuth raw range: [{azimuth_raw.min():.2f}, {azimuth_raw.max():.2f}]")
print(f"Azimuth corrected range: [{azimuth_corrected.min():.2f}, {azimuth_corrected.max():.2f}]")
print(f"Polar raw range: [{polar_raw.min():.2f}, {polar_raw.max():.2f}]")
print(f"Polar corrected range: [{polar_corrected.min():.2f}, {polar_corrected.max():.2f}]")

az_diff_raw = np.diff(azimuth_raw)
az_diff_corr = np.diff(azimuth_corrected)
po_diff_raw = np.diff(polar_raw)
po_diff_corr = np.diff(polar_corrected)

az_jumps_raw = np.sum(np.abs(az_diff_raw) > 180)
az_jumps_corr = np.sum(np.abs(az_diff_corr) > 180)
po_jumps_raw = np.sum(np.abs(po_diff_raw) > 90)
po_jumps_corr = np.sum(np.abs(po_diff_corr) > 90)

print(f"\nAzimuth large jumps (>180°): raw {az_jumps_raw} occurrences -> corrected {az_jumps_corr} occurrences")
print(f"Polar large jumps (>90°): raw {po_jumps_raw} occurrences -> corrected {po_jumps_corr} occurrences")

# Save results
result_df = pd.DataFrame({
    'time': time,
    'azimuth_raw': azimuth_raw,
    'azimuth_corrected': azimuth_corrected,
    'polar_raw': polar_raw,
    'polar_corrected': polar_corrected
})
result_df.to_excel(f'corrected_angles_neighbor_window{window_size}.xlsx', index=False)
print(f"\nResults saved to corrected_angles_neighbor_window{window_size}.xlsx")