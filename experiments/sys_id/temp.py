
#
# for lvl in [1, 2, 3, 4, 5]:
#     coeffs = pywt.wavedec(EEG_down, wavelet='db4', level=lvl)
#     print("Level", lvl, "→", sum(map(len, coeffs)), "coeffs")
#
#
# for w in ['haar','db2','db4','sym5']:
#     coeffs = pywt.wavedec(EEG_down, wavelet=w, level=1, mode='periodization')
#     print(w, "→", sum(len(c) for c in coeffs), "coeffs")
