import numpy as np

# Generate some synthetic data for analysis
data = np.random.randn(10000, 10)  # 10,000 samples, 10 features each
labels = (data[:, 0] > 0).astype(int)  # Binary labels based on the first feature

# Normalize the data
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data = (data - data_mean) / data_std

# Save preprocessed data
np.save('data.npy', data)
np.save('labels.npy', labels)
