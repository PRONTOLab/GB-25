import re
import numpy as np
import matplotlib.pyplot as plt

data = []
with open("/tmp/r0_timeseries.txt") as f:
    for line in f:
        m = re.match(
            r'iter\s+(\d+)\s+t=\s*([\d.]+)s\s+wall=\s*([\d.]+)s\s+'
            r'ρ=\[([^,]+),([^\]]+)\]\s+'
            r'ρu=\[([^,]+),([^\]]+)\]\s+'
            r'ρv=\[([^,]+),([^\]]+)\]\s+'
            r'ρw=\[([^,]+),([^\]]+)\]\s+'
            r'ρθ=\[([^,]+),([^\]]+)\]\s+'
            r'ρqv=\[([^,]+),([^\]]+)\]',
            line.strip()
        )
        if m:
            data.append([float(x) for x in m.groups()])

d = np.array(data)
iters = d[:, 0]
sim_time_h = d[:, 1] / 3600  # hours
wall_h = d[:, 2] / 3600

rho_min, rho_max = d[:, 3], d[:, 4]
ru_min, ru_max = d[:, 5], d[:, 6]
rv_min, rv_max = d[:, 7], d[:, 8]
rw_min, rw_max = d[:, 9], d[:, 10]
rt_min, rt_max = d[:, 11], d[:, 12]
rqv_min, rqv_max = d[:, 13], d[:, 14]

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle("1/24° NCCL 8×H100 production run — rank 0 field extrema\n"
             "Δt=0.8s, τ_cloud=120s, cloud-field IC, qv clamp", fontsize=13)

def plot_range(ax, t, lo, hi, label, color='C0'):
    ax.fill_between(t, lo, hi, alpha=0.3, color=color)
    ax.plot(t, lo, color=color, lw=0.8)
    ax.plot(t, hi, color=color, lw=0.8)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)

plot_range(axes[0, 0], sim_time_h, rho_min, rho_max, r'$\rho$ [kg/m³]', 'C0')
plot_range(axes[0, 1], sim_time_h, ru_min, ru_max, r'$\rho u$ [kg/m²s]', 'C1')
plot_range(axes[1, 0], sim_time_h, rv_min, rv_max, r'$\rho v$ [kg/m²s]', 'C2')
plot_range(axes[1, 1], sim_time_h, rw_min, rw_max, r'$\rho w$ [kg/m²s]', 'C3')
plot_range(axes[2, 0], sim_time_h, rt_min, rt_max, r'$\rho\theta$ [kg K/m³]', 'C4')
plot_range(axes[2, 1], sim_time_h, rqv_min, rqv_max, r'$\rho q_v$ [kg/m³]', 'C5')

for ax in axes[-1, :]:
    ax.set_xlabel('Simulation time [hours]')

plt.tight_layout()
plt.savefig('/teamspace/studios/this_studio/GB-25/diagnostics_timeseries.png', dpi=150)
print("Saved diagnostics_timeseries.png")
