import numpy as np
import matplotlib.pyplot as plt

# 1. Standard Setup
x = np.arange(256).reshape(1, 256)
y = np.arange(256).reshape(256, 1)

# RED: Vertical Stripes
R = (128 + 127 * np.sin(x / 8) + 0 * y).astype(np.uint8)

# GREEN: Horizontal Stripes
# We use a different frequency so they don't perfectly align with Red.
G = (128 + 127 * np.sin(y / 12) + 0 * x).astype(np.uint8)

# BLUE
B = (128 + 127 * np.sin((x + y) / 10)).astype(np.uint8)

# 3. Combine and Plot
rgb = np.dstack((R, G, B)).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(rgb)
plt.title("Broadcasting Art: Geometric Weave")
plt.axis('off')
plt.show()