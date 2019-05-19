import numpy as np
import matplotlib.pyplot as plt

end = (2.0*np.pi - 87)/18
start = -87/18.0

print("采样起点：%f，采样终点：%f"%(start, end))

line = np.linspace(start, end, 2000, dtype=float)
y = np.cos(line*18+87)

plt.plot(line, y, 'g-', linewidth=1)

poly = np.polyfit(line, y, deg=3)
z = np.polyval(poly, line)
print(poly)
plt.plot(line, z, 'r-', linewidth=1)

plt.show()

print(len(line))