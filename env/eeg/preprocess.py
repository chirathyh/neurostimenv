import scipy.signal as ss

def eeg_pre_processing(self, data, filter_data=True, ztransform=True):
        # # data = ss.decimate(data, q=16, zero_phase=True)  # down sample?
        if filter_data:
            # b, a = ss.butter(N=2, Wn=0.02, btype='lowpass')
            b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=self.sampling_rate, output='ba')
            data = ss.filtfilt(b, a, data, axis=-1)
        # if ztransform:
        #     # substract the mean of a channel
        #     dataT = data.T - data.mean(axis=1)
        #     data = dataT.T
        return data
