"""
L12 Implementation: Generate and Visualize Overlapping 2D Data Groups

This program generates three groups of 2D data points with 20% overlap
and visualizes them using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def generate_overlapping_groups(n_points=2000, overlap_percentage=0.20, random_seed=42):
    """
    Generate three groups of 2D data points with specified overlap.

    Parameters:
    -----------
    n_points : int
        Total number of points per group (default: 2000)
    overlap_percentage : float
        Percentage of points that overlap between groups (default: 0.20)
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    group1, group2, group3 : numpy arrays
        Three arrays of shape (n_points, 2) containing 2D coordinates
    """
    np.random.seed(random_seed)

    # Calculate number of unique and overlapping points
    n_overlap = int(n_points * overlap_percentage)
    n_unique = n_points - n_overlap

    # Define cluster centers for each group
    center1 = np.array([0, 0])
    center2 = np.array([3, 2])
    center3 = np.array([1, 4])

    # Standard deviation for cluster spread
    std_dev = 1.0

    # Generate unique points for each group (80% of each group)
    unique1 = np.random.randn(n_unique, 2) * std_dev + center1
    unique2 = np.random.randn(n_unique, 2) * std_dev + center2
    unique3 = np.random.randn(n_unique, 2) * std_dev + center3

    # Generate overlapping points in regions between clusters
    # Split overlap points into different overlap regions
    n_overlap_12 = n_overlap // 3  # Overlap between group 1 and 2
    n_overlap_23 = n_overlap // 3  # Overlap between group 2 and 3
    n_overlap_13 = n_overlap - n_overlap_12 - n_overlap_23  # Overlap between 1 and 3

    # Generate overlap points between groups 1 and 2
    center_12 = (center1 + center2) / 2
    overlap_12 = np.random.randn(n_overlap_12, 2) * (std_dev * 0.8) + center_12

    # Generate overlap points between groups 2 and 3
    center_23 = (center2 + center3) / 2
    overlap_23 = np.random.randn(n_overlap_23, 2) * (std_dev * 0.8) + center_23

    # Generate overlap points between groups 1 and 3
    center_13 = (center1 + center3) / 2
    overlap_13 = np.random.randn(n_overlap_13, 2) * (std_dev * 0.8) + center_13

    # Combine all overlapping points
    overlap_all = np.vstack([overlap_12, overlap_23, overlap_13])

    # Create complete groups by combining unique and overlapping points
    group1 = np.vstack([unique1, overlap_12, overlap_13])
    group2 = np.vstack([unique2, overlap_12, overlap_23])
    group3 = np.vstack([unique3, overlap_23, overlap_13])

    return group1, group2, group3, overlap_all


def kmeans_clustering(data, k=3, max_iterations=5):
    """
    Perform k-means clustering on the data.

    Parameters:
    -----------
    data : numpy array
        Array of shape (n_samples, 2) containing 2D coordinates
    k : int
        Number of clusters (default: 3)
    max_iterations : int
        Maximum number of iterations (default: 5)

    Returns:
    --------
    centroids : numpy array
        Final centroid positions of shape (k, 2)
    labels : numpy array
        Cluster assignment for each data point
    """
    # Initialize centroids randomly from the data points
    np.random.seed(42)
    random_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[random_indices].copy()

    print(f"\nRunning K-means with K={k} for {max_iterations} iterations...")

    for iteration in range(max_iterations):
        # Assign each point to the nearest centroid
        distances = np.zeros((len(data), k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((data - centroids[i])**2, axis=1))

        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        centroids = new_centroids
        print(f"  Iteration {iteration + 1}/{max_iterations} completed")

    print("K-means clustering completed!\n")
    return centroids, labels


def visualize_groups(group1, group2, group3, overlap_points):
    """
    Visualize the three groups of data points with interactive K-means button.

    Parameters:
    -----------
    group1, group2, group3 : numpy arrays
        Arrays containing 2D coordinates for each group
    overlap_points : numpy array
        Array containing overlapping points
    """
    # Combine all data points into a single array
    all_data = np.vstack([group1, group2, group3])

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)  # Make room for button

    # Initial plot - all groups with transparency to show overlap
    scatter1 = ax.scatter(group1[:, 0], group1[:, 1], c='red', alpha=0.4, s=20, label='Group 1')
    scatter2 = ax.scatter(group2[:, 0], group2[:, 1], c='blue', alpha=0.4, s=20, label='Group 2')
    scatter3 = ax.scatter(group3[:, 0], group3[:, 1], c='green', alpha=0.4, s=20, label='Group 3')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Three Overlapping Groups (2000 points each, 20% overlap)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Store references for updating
    state = {
        'original_scatters': [scatter1, scatter2, scatter3],
        'kmeans_scatter': None,
        'centroids_scatter': None,
        'showing_kmeans': False,
        'centroids': None,
        'labels': None,
        'scattered': False
    }

    def on_kmeans_button_click(event):
        """Callback function for K-means button."""
        if not state['showing_kmeans']:
            # Run K-means clustering
            centroids, labels = kmeans_clustering(all_data, k=3, max_iterations=5)

            # Store centroids and labels in state
            state['centroids'] = centroids
            state['labels'] = labels
            state['scattered'] = False

            # Hide original groups
            for scatter in state['original_scatters']:
                scatter.set_visible(False)

            # Create original color array (red for group1, blue for group2, green for group3)
            n1, n2, n3 = len(group1), len(group2), len(group3)
            original_colors = np.array(['red'] * n1 + ['blue'] * n2 + ['green'] * n3)

            # Cluster edge colors
            cluster_edge_colors = ['purple', 'orange', 'cyan']

            # Plot points with original fill color and cluster edge color
            edge_colors = np.array([cluster_edge_colors[label] for label in labels])

            state['kmeans_scatter'] = ax.scatter(all_data[:, 0], all_data[:, 1],
                                                  c=original_colors,
                                                  edgecolors=edge_colors,
                                                  alpha=0.6, s=40, linewidths=2,
                                                  label='Clustered Points')

            # Plot centroids
            state['centroids_scatter'] = ax.scatter(centroids[:, 0], centroids[:, 1],
                                                     c='black', marker='X', s=200,
                                                     edgecolors='white', linewidths=2,
                                                     label='Centroids', zorder=5)

            # Create custom legend for clusters
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='none', edgecolor='purple', label='Cluster 1', linewidth=2),
                Patch(facecolor='none', edgecolor='orange', label='Cluster 2', linewidth=2),
                Patch(facecolor='none', edgecolor='cyan', label='Cluster 3', linewidth=2),
                plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                          markersize=10, label='Centroids', markeredgecolor='white', markeredgewidth=1)
            ]
            ax.legend(handles=legend_elements)

            ax.set_title('K-means Clustering Results (K=3, 5 iterations)')
            state['showing_kmeans'] = True
            button.label.set_text('Show Original')
        else:
            # Clear K-means results
            for collection in ax.collections[3:]:  # Remove K-means scatters and centroids
                collection.remove()

            # Show original groups again
            for scatter in state['original_scatters']:
                scatter.set_visible(True)

            ax.set_title('Three Overlapping Groups (2000 points each, 20% overlap)')
            ax.legend()
            state['showing_kmeans'] = False
            state['centroids'] = None
            state['labels'] = None
            button.label.set_text('Run K-means')

        plt.draw()

    def on_click(event):
        """Handle mouse click events to detect clicks near centroids."""
        # Only handle clicks when K-means is showing
        if not state['showing_kmeans'] or state['centroids'] is None:
            return

        # Ignore clicks outside the main axes
        if event.inaxes != ax:
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        # Check if click is near any centroid (within threshold distance)
        threshold = 0.5  # Distance threshold for detecting click near centroid
        centroids = state['centroids']
        labels = state['labels']

        for i, centroid in enumerate(centroids):
            distance = np.sqrt((click_x - centroid[0])**2 + (click_y - centroid[1])**2)
            if distance < threshold:
                print(f"\nClick detected near Cluster {i+1} centroid!")
                print(f"Scattering Cluster {i+1} points randomly...")

                # Get current axis limits to scatter within view
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                # Get current positions
                current_positions = state['kmeans_scatter'].get_offsets()
                new_positions = current_positions.copy()

                # Only scatter points belonging to this cluster
                cluster_mask = labels == i
                n_cluster_points = np.sum(cluster_mask)

                # Generate random positions for this cluster's points only
                random_x = np.random.uniform(x_min, x_max, n_cluster_points)
                random_y = np.random.uniform(y_min, y_max, n_cluster_points)

                # Update only the positions of points in this cluster
                new_positions[cluster_mask] = np.c_[random_x, random_y]

                # Update the scatter plot with new positions
                if state['kmeans_scatter'] is not None:
                    state['kmeans_scatter'].set_offsets(new_positions)

                # Keep centroids in place
                state['scattered'] = True
                ax.set_title(f'Cluster {i+1} Points Randomly Scattered!')
                plt.draw()
                break

    # Add button for K-means clustering
    button_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(button_ax, 'Run K-means')
    button.on_clicked(on_kmeans_button_click)

    # Connect click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.savefig('overlapping_groups.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'overlapping_groups.png'")
    plt.show()


def print_statistics(group1, group2, group3):
    """
    Print statistics about the generated groups.

    Parameters:
    -----------
    group1, group2, group3 : numpy arrays
        Arrays containing 2D coordinates for each group
    """
    print("\n" + "="*60)
    print("Data Generation Statistics")
    print("="*60)
    print(f"Group 1 size: {len(group1)} points")
    print(f"Group 2 size: {len(group2)} points")
    print(f"Group 3 size: {len(group3)} points")
    print(f"\nGroup 1 - X range: [{group1[:, 0].min():.2f}, {group1[:, 0].max():.2f}]")
    print(f"Group 1 - Y range: [{group1[:, 1].min():.2f}, {group1[:, 1].max():.2f}]")
    print(f"\nGroup 2 - X range: [{group2[:, 0].min():.2f}, {group2[:, 0].max():.2f}]")
    print(f"Group 2 - Y range: [{group2[:, 1].min():.2f}, {group2[:, 1].max():.2f}]")
    print(f"\nGroup 3 - X range: [{group3[:, 0].min():.2f}, {group3[:, 0].max():.2f}]")
    print(f"Group 3 - Y range: [{group3[:, 1].min():.2f}, {group3[:, 1].max():.2f}]")
    print("="*60 + "\n")


def main():
    """
    Main function to generate and visualize overlapping data groups.
    """
    print("Generating three groups of 2D data points with 20% overlap...")

    # Generate the groups
    group1, group2, group3, overlap_points = generate_overlapping_groups(
        n_points=2000,
        overlap_percentage=0.20,
        random_seed=42
    )

    # Print statistics
    print_statistics(group1, group2, group3)

    # Visualize the groups
    print("Creating visualization...")
    visualize_groups(group1, group2, group3, overlap_points)

    print("\nDone!")


if __name__ == "__main__":
    main()
