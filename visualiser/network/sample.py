import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apply a Seaborn theme for a polished grid background
sns.set_style('whitegrid')

# Create dummy “power spectra”
freqs = np.linspace(5, 30, 200)
healthy_mean = 1.5 + np.exp(-0.5*((freqs - 10)/2)**2)
depress_mean = healthy_mean + 0.3 * np.sin((freqs - 5) / 5)
healthy_std = 0.1 * np.ones_like(freqs)
depress_std = 0.1 * np.ones_like(freqs)

plt.figure(figsize=(6, 4))

# Plot healthy
plt.plot(freqs, healthy_mean, color='k', lw=2, label='Healthy')
plt.fill_between(freqs,
                 healthy_mean - healthy_std,
                 healthy_mean + healthy_std,
                 color='k', alpha=0.2)

# Plot depression
plt.plot(freqs, depress_mean, color='r', lw=2, label='Depression')
plt.fill_between(freqs,
                 depress_mean - depress_std,
                 depress_mean + depress_std,
                 color='r', alpha=0.2)

# Mark θ and α band boundaries
plt.axvline(8, color='gray', linestyle='--')
plt.axvline(12, color='gray', linestyle='--')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.legend()
plt.tight_layout()
plt.show()
