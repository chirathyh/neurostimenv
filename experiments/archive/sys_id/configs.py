
def get_configs():
    dt = 0.025
    fs = (1 / dt) * 1000

    nperseg = int(fs/2)
    noverlap = int(nperseg*0.5) # nperseg-1 #nperseg // 2

    transient = 4000  # ms; first 4s is removed from the EEG (triansient phase)
    t1 = int(transient/dt)

    print("Sampling Rate:", fs)
    print("npserg", nperseg)

    return dt, fs, nperseg, noverlap, t1
