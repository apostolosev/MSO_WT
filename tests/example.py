import numpy as np
import matplotlib.pyplot as plt
from mso_wt import MSOWT

# Define the example multi-component input sign
t = np.linspace(0.0, 1.0, 1024)
gt_imf1 = 1 / (1.2 + np.cos(2 * np.pi * t))
gt_imf2 = np.cos(32 * np.pi * t + 0.2 * np.cos(64 * np.pi * t)) / (1.5 + np.sin(2 * np.pi * t))
sig = gt_imf1 + gt_imf2

# Decompose the signal using the MSO-WT
msowt = MSOWT(nIMFs=2)
imfs = msowt.decompose(sig)

# Display the decomposition results
# Original signal
plt.figure(figsize=(6, 4))
plt.plot(t, sig, "-k", linewidth=1.5, label="GT-IMF_1 + GT-IMF_2")
plt.title("Multi-component signal")
plt.legend(loc="upper left")
plt.xlabel("Time (s)")

# IMF 1
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(t, gt_imf1, "-k", linewidth=1.5, label="GT-IMF_1")
axs[0].plot(t, imfs[0, :], "-c", linewidth=1.5, label="IMF_1")
axs[0].legend(loc="upper left")
axs[0].set_xlabel("Time (s)")

# IMF 2
axs[1].plot(t, gt_imf2, "-k", linewidth=1.5, label="GT-IMF_2")
axs[1].plot(t, imfs[1, :], "-c", linewidth=1.5, label="IMF_2")
axs[1].legend(loc="upper left")
axs[1].set_xlabel("Time (s)")
plt.show()

