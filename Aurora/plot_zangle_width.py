import matplotlib.pyplot as plt
import numpy as np
# save_zangle_width_file = '/home/ljm/NiuChuang/AuroraObjectData/zangle_width/agw_tr1058_te38044_arc_line (copy 1).txt'
save_zangle_width_file = '/home/ljm/NiuChuang/AuroraObjectData/zangle_width/agw_tr1058_te38044_arc_cnd2_line.txt'
f = open(save_zangle_width_file, 'r')
lines = f.readlines()
num_arc = len(lines)

zenith_angles = []
arc_widths = []
for a in range(num_arc):
    line = lines[a]
    angle = float(line.split()[0])
    width = float(line.split()[1][:-1])

    zenith_angles.append(angle)
    arc_widths.append(width)

plot_size_h = 6
plot_size_w = 8
fig_id = 1
plt.figure(fig_id, figsize=[plot_size_w, plot_size_h])
fig_id += 1
plt.scatter(zenith_angles, arc_widths, s=2)
plt.title("Zenith angle range: -90~90")

zenith_angles = np.array(zenith_angles)
arc_widths = np.array(arc_widths)

thresh_a = 45
thresh_w = 100
index_a = np.abs(zenith_angles) <= thresh_a
index_w = arc_widths <= thresh_w

index = index_a * index_w

zenith_angles_s = zenith_angles[index]
arc_widths_s = arc_widths[index]

plt.figure(fig_id, figsize=[plot_size_w, plot_size_h])
fig_id += 1
plt.scatter(zenith_angles_s, arc_widths_s, s=4, c='g')
# plt.title("Zenith angle range: -{}~{}".format(thresh_a, thresh_a))
plt.ylabel('Width (km)')
plt.xlabel('Zenith angle')

# mean curve.
angle_range = list(range(-thresh_a, thresh_a+1))

# zenith_angles_s_int = np.int(zenith_angles_s)
arc_widths_s_mean = np.zeros((len(angle_range)))

for a in range(len(angle_range)):
    angle = angle_range[a]
    index_l = zenith_angles_s >= angle
    index_r = zenith_angles_s < angle+1
    index = index_l * index_r
    arc_widths_s_a = arc_widths_s[index]

    arc_widths_s_mean[a] = arc_widths_s_a.mean()
    # arc_widths_s_mean[a] = (arc_widths_s_a.max() + arc_widths_s_a.min()) / 2

plt.plot(angle_range, arc_widths_s_mean, c='b')
mean_point = -8.9
print("mean zenith angle:", mean_point)
plt.plot([mean_point, mean_point], [0, thresh_w], linestyle='--', linewidth=3, color='blue')
plt.savefig('width_distribution_cnd2.png', dpi=300, bbox_inches='tight', transparent=True)

# Compute the mean and standard deviation.
thresh_a = 15
index_ss_r = zenith_angles_s <= mean_point + thresh_a
index_ss_l = zenith_angles_s >= mean_point - thresh_a
index_ss = index_ss_l*index_ss_r

zenith_angles_ss = zenith_angles_s[index_ss]
arc_widths_ss = arc_widths_s[index_ss]

arc_ss_mean = arc_widths_ss.mean()
arc_ss_std = np.std(arc_widths_ss, ddof=1)
print("mean:", arc_ss_mean)
print("std::", arc_ss_std)

plt.show()