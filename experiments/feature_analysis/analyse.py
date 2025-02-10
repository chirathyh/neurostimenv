import numpy as np

# Load CSV file
data = np.loadtxt("data/EEG0.csv", delimiter=",")

print("Loaded data:\n", data)
