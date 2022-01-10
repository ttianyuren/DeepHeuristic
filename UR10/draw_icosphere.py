from matplotlib import pyplot as plt
from mesh.mesh import icosphere


def color_fn(i):
    if i < 12:
        return 'navy'
    elif i < 42:
        return 'firebrick'
    elif i < 162:
        return 'orange'
    elif i < 642:
        return 'firebrick'
    else:
        return 'lightseagreen'


s = icosphere(2)

g = s.graphs_by_level[-1]

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# x, y, z = [1, 1.5], [1, 2.4], [3.4, 1.4]
# ax.scatter(x, y, z, c='red', s=100)
# ax.plot(x, y, z, color='black')
# x, y, z = [2.5, 1], [3.5, 1], [2, 3.4]
# ax.scatter(x, y, z, c='red', s=100)
# ax.plot(x, y, z, color='black')

for e in g.edges:
    v1, v2 = e
    p1, p2 = s.vertices[v1], s.vertices[v2]
    x, y, z = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
    ax.plot(x, y, z, color='dimgray')

for i, p in enumerate(s.vertices):
    x, y, z = p
    ax.scatter(x, y, z, c=color_fn(i), s=100)

ax.set_box_aspect([1, 1, 1])

plt.axis('off')
plt.show()
