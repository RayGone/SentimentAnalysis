import numpy as np
depth = 200
length = 200


positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
depths = np.arange(depth)[np.newaxis, :]/(depth/2)  # (1, depth)
print(depths.shape)
angle_rates = 1 / (10000**depths)         # (1, depth)
angle_rads = positions * angle_rates      # (pos, depth)

pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
print("Concatanation",str(pos_encoding[:2]),pos_encoding.shape)

ipos_encoding = np.zeros(pos_encoding.shape)
ipos_encoding[:, ::2] = np.cos(angle_rads)
ipos_encoding[:, 1::2] = np.sin(angle_rads)

print("Interleaving",str(ipos_encoding[:2]),ipos_encoding.shape)

# Check the shape.
import matplotlib.pyplot as plt

# Plot the dimensions.
plt.pcolormesh(pos_encoding.T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

plt.pcolormesh(ipos_encoding.T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

plt.pcolormesh((pos_encoding+ipos_encoding).T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()
