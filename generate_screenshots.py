"""
Screenshot Generation Script for README
Generates visualization screenshots at different stages for documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def generate_overlapping_groups(n_points=2000, overlap_percentage=0.20, random_seed=42):
    """Generate three groups of 2D data points with specified overlap."""
    np.random.seed(random_seed)

    n_overlap = int(n_points * overlap_percentage)
    n_unique = n_points - n_overlap

    center1 = np.array([0, 0])
    center2 = np.array([3, 2])
    center3 = np.array([1, 4])

    std_dev = 1.0

    unique1 = np.random.randn(n_unique, 2) * std_dev + center1
    unique2 = np.random.randn(n_unique, 2) * std_dev + center2
    unique3 = np.random.randn(n_unique, 2) * std_dev + center3

    n_overlap_12 = n_overlap // 3
    n_overlap_23 = n_overlap // 3
    n_overlap_13 = n_overlap - n_overlap_12 - n_overlap_23

    center_12 = (center1 + center2) / 2
    overlap_12 = np.random.randn(n_overlap_12, 2) * (std_dev * 0.8) + center_12

    center_23 = (center2 + center3) / 2
    overlap_23 = np.random.randn(n_overlap_23, 2) * (std_dev * 0.8) + center_23

    center_13 = (center1 + center3) / 2
    overlap_13 = np.random.randn(n_overlap_13, 2) * (std_dev * 0.8) + center_13

    group1 = np.vstack([unique1, overlap_12, overlap_13])
    group2 = np.vstack([unique2, overlap_12, overlap_23])
    group3 = np.vstack([unique3, overlap_23, overlap_13])

    return group1, group2, group3


def kmeans_clustering(data, k=3, max_iterations=100, tolerance=1e-6):
    """Perform k-means clustering with convergence detection."""
    np.random.seed(42)
    random_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[random_indices].copy()

    previous_labels = None

    for iteration in range(max_iterations):
        distances = np.zeros((len(data), k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((data - centroids[i])**2, axis=1))

        labels = np.argmin(distances, axis=1)

        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        centroid_shift = np.max(np.sqrt(np.sum((new_centroids - centroids)**2, axis=1)))
        centroids = new_centroids
        previous_labels = labels.copy()

        if centroid_shift < tolerance:
            break

    return centroids, labels


def screenshot_1_original_groups():
    """Screenshot 1: Original overlapping groups."""
    print("Generating Screenshot 1: Original Groups...")
    group1, group2, group3 = generate_overlapping_groups()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(group1[:, 0], group1[:, 1], c='red', alpha=0.4, s=20, label='Group 1')
    ax.scatter(group2[:, 0], group2[:, 1], c='blue', alpha=0.4, s=20, label='Group 2')
    ax.scatter(group3[:, 0], group3[:, 1], c='green', alpha=0.4, s=20, label='Group 3')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Three Overlapping Groups (2000 points each, 20% overlap)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/screenshots/01_original_groups.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: docs/screenshots/01_original_groups.png")


def screenshot_2_kmeans_result():
    """Screenshot 2: K-means clustering result."""
    print("Generating Screenshot 2: K-means Clustering Result...")
    group1, group2, group3 = generate_overlapping_groups()
    all_data = np.vstack([group1, group2, group3])

    centroids, labels = kmeans_clustering(all_data, k=3)

    n1, n2, n3 = len(group1), len(group2), len(group3)
    original_colors = np.array(['red'] * n1 + ['blue'] * n2 + ['green'] * n3)
    cluster_edge_colors = ['purple', 'orange', 'cyan']
    edge_colors = np.array([cluster_edge_colors[label] for label in labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(all_data[:, 0], all_data[:, 1], c=original_colors, edgecolors=edge_colors,
               alpha=0.6, s=40, linewidths=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
               edgecolors='white', linewidths=2, zorder=5)

    legend_elements = [
        Patch(facecolor='none', edgecolor='purple', label='Cluster 1', linewidth=2),
        Patch(facecolor='none', edgecolor='orange', label='Cluster 2', linewidth=2),
        Patch(facecolor='none', edgecolor='cyan', label='Cluster 3', linewidth=2),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                  markersize=10, label='Centroids', markeredgecolor='white', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('K-means Clustering Results (K=3, converged)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/screenshots/02_kmeans_clustered.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: docs/screenshots/02_kmeans_clustered.png")


def screenshot_3_after_scatter():
    """Screenshot 3: After scattering one cluster."""
    print("Generating Screenshot 3: After Scattering Cluster...")
    group1, group2, group3 = generate_overlapping_groups()
    all_data = np.vstack([group1, group2, group3])

    centroids, labels = kmeans_clustering(all_data, k=3)

    # Simulate scattering cluster 1
    cluster_to_scatter = 1
    cluster_mask = labels == cluster_to_scatter
    n_cluster_points = np.sum(cluster_mask)

    # Get current plot limits
    x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
    y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1

    # Scatter the cluster
    np.random.seed(100)  # Different seed for scattered points
    scattered_data = all_data.copy()
    scattered_data[cluster_mask, 0] = np.random.uniform(x_min, x_max, n_cluster_points)
    scattered_data[cluster_mask, 1] = np.random.uniform(y_min, y_max, n_cluster_points)

    n1, n2, n3 = len(group1), len(group2), len(group3)
    original_colors = np.array(['red'] * n1 + ['blue'] * n2 + ['green'] * n3)
    cluster_edge_colors = ['purple', 'orange', 'cyan']
    edge_colors = []

    for i, label in enumerate(labels):
        if label == cluster_to_scatter:
            edge_colors.append([0, 0, 0, 0])  # Transparent for scattered cluster
        else:
            edge_colors.append(cluster_edge_colors[label])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(scattered_data[:, 0], scattered_data[:, 1], c=original_colors, edgecolors=edge_colors,
               alpha=0.6, s=40, linewidths=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
               edgecolors='white', linewidths=2, zorder=5)

    legend_elements = [
        Patch(facecolor='none', edgecolor='purple', label='Cluster 1', linewidth=2),
        Patch(facecolor='none', edgecolor='orange', label='Cluster 2 (Scattered)', linewidth=2),
        Patch(facecolor='none', edgecolor='cyan', label='Cluster 3', linewidth=2),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                  markersize=10, label='Centroids', markeredgecolor='white', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Cluster 2 Points Randomly Scattered!', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/screenshots/03_after_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: docs/screenshots/03_after_scatter.png")


def screenshot_4_after_drag():
    """Screenshot 4: After dragging a cluster."""
    print("Generating Screenshot 4: After Dragging Cluster...")
    group1, group2, group3 = generate_overlapping_groups()
    all_data = np.vstack([group1, group2, group3])

    centroids, labels = kmeans_clustering(all_data, k=3)

    # Simulate dragging cluster 0
    cluster_to_drag = 0
    cluster_mask = labels == cluster_to_drag
    drag_offset = np.array([2.5, -1.5])

    # Move the cluster
    dragged_data = all_data.copy()
    dragged_data[cluster_mask] += drag_offset
    dragged_centroids = centroids.copy()
    dragged_centroids[cluster_to_drag] += drag_offset

    n1, n2, n3 = len(group1), len(group2), len(group3)
    original_colors = np.array(['red'] * n1 + ['blue'] * n2 + ['green'] * n3)
    cluster_edge_colors = ['purple', 'orange', 'cyan']
    edge_colors = np.array([cluster_edge_colors[label] for label in labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(dragged_data[:, 0], dragged_data[:, 1], c=original_colors, edgecolors=edge_colors,
               alpha=0.6, s=40, linewidths=2)
    ax.scatter(dragged_centroids[:, 0], dragged_centroids[:, 1], c='black', marker='X', s=200,
               edgecolors='white', linewidths=2, zorder=5)

    # Draw arrow showing the drag
    arrow_start = centroids[cluster_to_drag]
    arrow_end = dragged_centroids[cluster_to_drag]
    ax.annotate('', xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))

    legend_elements = [
        Patch(facecolor='none', edgecolor='purple', label='Cluster 1 (Dragged)', linewidth=2),
        Patch(facecolor='none', edgecolor='orange', label='Cluster 2', linewidth=2),
        Patch(facecolor='none', edgecolor='cyan', label='Cluster 3', linewidth=2),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                  markersize=10, label='Centroids', markeredgecolor='white', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Cluster 1 Dragged to New Location', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/screenshots/04_after_drag.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: docs/screenshots/04_after_drag.png")


def main():
    """Generate all screenshots."""
    import os

    # Create screenshots directory
    os.makedirs('docs/screenshots', exist_ok=True)

    print("\n" + "="*60)
    print("Screenshot Generation for README")
    print("="*60 + "\n")

    screenshot_1_original_groups()
    screenshot_2_kmeans_result()
    screenshot_3_after_scatter()
    screenshot_4_after_drag()

    print("\n" + "="*60)
    print("[SUCCESS] All screenshots generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
