import json
import matplotlib.pyplot as plt

path = r'point_ray_search_benchmark/benchmark.json'
with open(path, 'r') as f:
    lines = f.readlines()

uniform_grid = []
kdtree = []
octree = []
ours = []
rays = []

for line in lines:
    line = json.loads(line)
    rays.append(line['num_ray'])
    uniform_grid.append(line['uniform_grid']['build'][0] + line['uniform_grid']['search'][0])
    kdtree.append(line['kdtree']['build'][0] + line['kdtree']['search'][0])
    octree.append(line['octree']['build'][0] + line['octree']['search'][0])
    ours.append(line['plane']['build'][0] + line['plane']['search'][0])

plt.figure(figsize=(7, 5)) 

plt.plot(rays, uniform_grid, label='uniform_grid', linewidth=2)
plt.plot(rays, kdtree, label='kdtree', linewidth=2)
plt.plot(rays, octree, label='octree', linewidth=2)
plt.plot(rays, ours, label='ours', linewidth=2)


plt.legend(loc='upper left', fontsize=12)
plt.xlabel('number of rays', fontsize=14)
plt.ylabel('time (s)', fontsize=14)
plt.title('Benchmark Results', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig('point_ray_search_benchmark/benchmark.png', dpi=300, bbox_inches='tight', pad_inches=0)

path = r'point_ray_search_benchmark/benchmark1.json'
with open(path, 'r') as f:
    lines = f.readlines()

uniform_grid = []
kdtree = []
octree = []
ours = []
num_point = []

for line in lines:
    line = json.loads(line)
    num_point.append(line['num_point'])
    uniform_grid.append(line['uniform_grid']['build'][0] + line['uniform_grid']['search'][0])
    kdtree.append(line['kdtree']['build'][0] + line['kdtree']['search'][0])
    octree.append(line['octree']['build'][0] + line['octree']['search'][0])
    ours.append(line['plane']['build'][0] + line['plane']['search'][0])

plt.figure(figsize=(7, 5)) 

plt.plot(num_point, uniform_grid, label='uniform_grid', linewidth=2)
plt.plot(num_point, kdtree, label='kdtree', linewidth=2)
plt.plot(num_point, octree, label='octree', linewidth=2)
plt.plot(num_point, ours, label='ours', linewidth=2)


plt.legend(loc='upper left', fontsize=12)
plt.xlabel('number of points', fontsize=14)
plt.ylabel('time (s)', fontsize=14)
plt.title('Benchmark Results', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig('point_ray_search_benchmark/benchmark1.png', dpi=300, bbox_inches='tight', pad_inches=0)
