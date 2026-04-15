"""Plot all fields at surface from assembled 1/16° checkpoint."""
import numpy as np
import matplotlib.pyplot as plt
import h5py

path = "simulations/initial_conditions/sixteenth_deg_12h_assembled.jld2"
f = h5py.File(path, "r")

fields = {}
for name in ["ρ", "ρu", "ρv", "ρw", "ρθ", "ρqᵛ", "micro_ρqᶜˡ", "micro_ρqᶜⁱ", "ρqʳ"]:
    fields[name] = np.array(f[name]).T  # Julia col-major → C row-major
f.close()

Nλ = fields["ρ"].shape[0]
Nφ = fields["ρ"].shape[1]
lon = np.linspace(0, 360, Nλ, endpoint=False)
lat = np.linspace(-80, 80, Nφ, endpoint=False)

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
fig.suptitle("Assembled 1/16° checkpoint (12h) — all fields at surface (k=0)", fontsize=14)

configs = [
    ("ρ", fields["ρ"][:,:,0], "RdYlBu_r", None),
    ("ρu", fields["ρu"][:,:,0], "RdBu_r", "sym"),
    ("ρv", fields["ρv"][:,:Nφ,0], "RdBu_r", "sym"),  # trim face
    ("ρw", fields["ρw"][:,:,0], "RdBu_r", "sym"),
    ("ρθ", fields["ρθ"][:,:,0], "RdYlBu_r", None),
    ("ρqᵛ", fields["ρqᵛ"][:,:,0] * 1000, "YlGnBu", None),
    ("ρqᶜˡ", fields["micro_ρqᶜˡ"][:,:,0] * 1000, "Blues", None),
    ("ρqᶜⁱ", fields["micro_ρqᶜⁱ"][:,:,0] * 1000, "Purples", None),
    ("ρqʳ", fields["ρqʳ"][:,:,0] * 1000, "Oranges", None),
]

for idx, (name, data, cmap, mode) in enumerate(configs):
    ax = axes[idx // 3, idx % 3]
    d = data.T  # (lon, lat) → (lat, lon)
    if mode == "sym":
        vmax = np.percentile(np.abs(d), 99)
        im = ax.pcolormesh(lon, lat[:d.shape[0]], d, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='nearest', rasterized=True)
    else:
        vmin, vmax = np.percentile(d, [1, 99])
        im = ax.pcolormesh(lon, lat[:d.shape[0]], d, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='nearest', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(name)
    ax.set_ylabel('lat')
    if idx >= 6:
        ax.set_xlabel('lon')

plt.tight_layout()
plt.savefig('assembled_surface_fields.png', dpi=150, bbox_inches='tight')
print("Saved assembled_surface_fields.png")
