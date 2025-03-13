import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import scipy.signal as ss

dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # in seconds L23Net uses : 2000
t1 = int(transient/dt)
EEG = np.loadtxt("../../data/bandit/nbandit2/testing/EEG_BANDIT_1060.csv", delimiter=",")
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
x = ss.filtfilt(b, a, EEG[t1:], axis=-1)

T_x = 1/fs

N = len(x)  # 20 Hz sampling rate for 50 s signal
t_x = np.arange(N) * T_x  # time indexes for signal


# f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
# x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal


g_std = 8  # standard deviation for Gaussian window in samples
w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
SFT = ShortTimeFFT(w, hop=int(fs/4), fs=1/T_x, mfft=50, scale_to='magnitude')
Sx = SFT.stft(x)  # perform the STFT

instantaneous_phase = np.angle(Sx)  # Extract phase from STFT

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
              rf"$\sigma_t={g_std*SFT.T}\,$s)")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))
im1 = ax1.imshow(instantaneous_phase, origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='viridis')
# ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
fig1.colorbar(im1, label="Phase (radians)")

# Shade areas where window slices stick out to the side:
for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                 (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
for t_ in [0, N * SFT.T]:  # Mark signal borders with vertical line:
    ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)

ax1.legend()
ax1.set_ylim(0, 50)
fig1.tight_layout()
plt.show()



# im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
#                  extent=SFT.extent(N), cmap='viridis')
# ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
# fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
# # Shade areas where window slices stick out to the side:
# for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
#                  (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
#     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
# for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
#     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
# ax1.legend()
# fig1.tight_layout()
# plt.show()
