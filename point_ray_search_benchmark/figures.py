from point_ray_search_benchmark.uniform_grid import *
from point_ray_search_benchmark.kdtree import *
from point_ray_search_benchmark.quadtree import *

def main():
    np.random.seed(0)
    points = np.random.rand(100, 2) * 100  

    voxel_size = np.array([25, 25])
    uniform_grid = UniformGrid(points, voxel_size)

    ray_origin = np.array([0, 75])
    ray_direction = np.array([1, -0.5])
    # ray_direction = ray_direction / np.linalg.norm(ray_direction)

    thresh = 5.0
    _, nearby_points = violent_enumeration_ray(ray_origin, ray_direction, points, thresh)
    nearby_point_indices = [point.name for point in nearby_points]
    
    
    # visualize uniform grid
    fig, ax = plt.subplots(figsize=(10,10))
    # visualize grid
    lines, colors = uniform_grid.draw_all_grids()
    lines = np.array(lines).reshape(-1, 4)
    # delete the repeated lines
    lines = np.unique(lines, axis=0)
    for line, color in zip(lines, colors):
        ax.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=2, zorder=1)

    # visualize all points 
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5, zorder=2)

    # visualize nearby points
    for point in nearby_points:
        ax.scatter(point.position[0], point.position[1], color='red', s=50, cmap='hsv', alpha=0.6, zorder=2)

    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    # plt.show()
    plt.savefig('point_ray_search_benchmark/uniform_grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    del lines

    # # visualize kdtree
    fig, ax = plt.subplots(figsize=(10,10))
    kdtree = KDTreePointSearch(points, max_pc_per_node=11)
    lines = kdtree.visualize_2d_kdtree(kdtree.root, min_x = 0, max_x = 100, min_y = 0, max_y = 100)
    lines.append([0, 100, 0, 0])
    lines.append([0, 0, 0, 100])
    lines.append([0, 100, 100, 100])
    lines.append([100, 100, 0, 100])
    color = 'black'
    for line in lines:
        ax.plot([line[0], line[1]], [line[2], line[3]], color=color, linewidth=2, zorder=1)
    
    # visualize all points 
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5, zorder=2)

    # visualize nearby points
    for point in nearby_points:
        ax.scatter(point.position[0], point.position[1], color='red', s=50, cmap='hsv', alpha=0.6, zorder=2)
    
    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    # plt.show()
    plt.savefig('point_ray_search_benchmark/kdtree.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    del lines

    # visualize octree
    fig, ax = plt.subplots(figsize=(10,10))
    quadtree = QuadTree(points,0, 100, 0, 100, 4)
    lines = list()
    quadtree.draw_all_grids(lines)
    color = 'black'
    for line in lines:
        ax.plot([line[0], line[1]], [line[2], line[3]], color=color, linewidth=2, zorder=1)
     # visualize all points 
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5, zorder=2)

    # visualize nearby points
    for point in nearby_points:
        ax.scatter(point.position[0], point.position[1], color='red', s=50, cmap='hsv', alpha=0.6, zorder=2)
    
    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    # plt.show()
    plt.savefig('point_ray_search_benchmark/quadtree.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    del lines

    # planes
    fig, ax = plt.subplots(figsize=(10,10))
    lines = list()
    lines.append([0, 100, 0, 0])
    lines.append([0, 0, 0, 100])
    lines.append([0, 100, 100, 100])
    lines.append([100, 100, 0, 100])
    color = 'black'
    for line in lines:
        ax.plot([line[0], line[1]], [line[2], line[3]], color=color, linewidth=2, zorder=1)
    # visualize all points 
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5, zorder=2)

    # visualize nearby points
    for point in nearby_points:
        ax.scatter(point.position[0], point.position[1], color='red', s=50, cmap='hsv', alpha=0.6, zorder=2)

    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    # plt.show()
    plt.savefig('point_ray_search_benchmark/plane.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    del lines

if __name__ == "__main__":
    main()