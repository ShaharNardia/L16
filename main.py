"""
L12 Implementation: Generate and Visualize Overlapping 2D Data Groups

This program generates three groups of 2D data points with 20% overlap
and visualizes them using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


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


def visualize_groups(group1, group2, group3, overlap_points):
    """
    Visualize the three groups of data points.

    Parameters:
    -----------
    group1, group2, group3 : numpy arrays
        Arrays containing 2D coordinates for each group
    overlap_points : numpy array
        Array containing overlapping points
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all groups with transparency to show overlap
    ax.scatter(group1[:, 0], group1[:, 1], c='red', alpha=0.4, s=20, label='Group 1')
    ax.scatter(group2[:, 0], group2[:, 1], c='blue', alpha=0.4, s=20, label='Group 2')
    ax.scatter(group3[:, 0], group3[:, 1], c='green', alpha=0.4, s=20, label='Group 3')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Three Overlapping Groups (2000 points each, 20% overlap)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
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
