# Requirements Document
## Interactive K-means Clustering Visualization System

---

### Document Information
- **Project Name**: Interactive K-means Clustering Visualization
- **Document Version**: 1.0
- **Last Updated**: 2025-11-02
- **Author**: Development Team

---

## 1. Overview

### 1.1 Purpose
This document specifies the requirements for a Python-based interactive visualization system that generates overlapping 2D data groups and performs K-means clustering with real-time user interaction capabilities.

### 1.2 Scope
The system shall provide:
- Synthetic data generation with controlled overlap
- Interactive K-means clustering visualization
- Real-time cluster manipulation through drag-and-drop
- Point scattering functionality for cluster analysis
- Visual feedback for clustering results

---

## 2. System Architecture

### 2.1 Technology Stack
- **Language**: Python
- **Libraries**:
  - NumPy (data generation and computation)
  - Matplotlib (visualization and interactive widgets)

### 2.2 Main Components
- Data generation module
- K-means clustering algorithm
- Interactive visualization interface
- Event handling system

---

## 3. Functional Requirements

### 3.1 Data Generation (FR-001)
**Requirement**: The system shall generate three groups of 2D data points.

**Specifications**:
- Each group contains exactly 2,000 points
- Groups shall have 20% overlap (approximately 400 points shared between groups)
- Data points shall be generated using NumPy
- Groups shall be centered around distinct cluster centers
- Overlap points shall be positioned in regions between cluster centers

**Priority**: High
**Source**: Prompt 1

---

### 3.2 Visualization (FR-002)

#### 3.2.1 Initial Display
**Requirement**: The system shall display the three overlapping data groups.

**Specifications**:
- Display a single plot showing all three groups
- Use distinct colors for each group (red, blue, green)
- Include transparency (alpha=0.4) to visualize overlaps
- Provide axis labels and title
- Include a grid for reference
- Include a legend identifying each group

**Priority**: High
**Source**: Prompts 1, 3

#### 3.2.2 Color Preservation
**Requirement**: Points shall maintain their original group colors throughout all operations.

**Specifications**:
- Original group colors (red, blue, green) persist after clustering
- Cluster assignments shown via edge colors, not fill colors
- Scattered points retain original fill colors

**Priority**: High
**Source**: Prompt 4

---

### 3.3 K-means Clustering (FR-003)

#### 3.3.1 Algorithm Execution
**Requirement**: The system shall implement K-means clustering with convergence detection.

**Specifications**:
- Use K=3 clusters
- Initialize centroids randomly from data points
- Implement convergence detection based on:
  - No points moving between clusters, OR
  - Centroids moving less than tolerance threshold (1e-6)
- Maximum iteration limit: 100 (safety measure)
- Display iteration progress in console
- Show convergence reason and iteration count

**Priority**: High
**Source**: Prompts 5, 10

#### 3.3.2 Cluster Visualization
**Requirement**: The system shall visualize K-means clustering results.

**Specifications**:
- Display points with original fill colors (red, blue, green)
- Add colored edge circles to indicate cluster assignments:
  - Cluster 1: Purple edge
  - Cluster 2: Orange edge
  - Cluster 3: Cyan edge
- Display centroids as black 'X' markers (size=200)
- Centroids have white edge lines for visibility
- Update plot title to "K-means Clustering Results (K=3, converged)"
- Update legend to show cluster colors and centroids

**Priority**: High
**Source**: Prompts 4, 5

---

### 3.4 User Interaction (FR-004)

#### 3.4.1 Run K-means Button
**Requirement**: The system shall provide a button to execute K-means clustering.

**Specifications**:
- Button labeled "Run K-means"
- Button always visible and enabled
- Position: Bottom center of figure
- First click: Run K-means on original data
- Subsequent clicks: Clear previous results and re-run K-means
- Hide original groups when K-means results are shown

**Priority**: High
**Source**: Prompts 5, 12

#### 3.4.2 Centroid Click to Scatter
**Requirement**: The system shall scatter cluster points when user clicks near a centroid.

**Specifications**:
- Detection threshold: 0.5 units from centroid
- Click-and-release without dragging triggers scatter
- Only scatter points belonging to the clicked cluster
- Random positions distributed across current axis limits
- Remove edge color (circle) from scattered points
- Update title to indicate which cluster was scattered
- Console feedback showing cluster number

**Priority**: Medium
**Source**: Prompts 6, 7, 9, 13

#### 3.4.3 Centroid Drag and Drop
**Requirement**: The system shall allow users to drag centroids and move entire clusters.

**Specifications**:
- Click and hold near centroid (within 0.5 units) to initiate drag
- Minimum drag threshold: 0.1 units to distinguish from click
- All points in cluster move with centroid
- Maintain relative positions of points within cluster
- Real-time visual feedback during drag
- Console feedback on drag start and completion
- Release mouse to complete drag operation

**Priority**: Medium
**Source**: Prompt 11

---

## 4. Non-Functional Requirements

### 4.1 Performance (NFR-001)
**Requirement**: The system shall provide responsive interaction.

**Specifications**:
- K-means clustering completes within reasonable time for 6,000 points
- Drag operations update visualization in real-time
- Scatter operations execute immediately upon click

**Priority**: Medium

### 4.2 Usability (NFR-002)
**Requirement**: The system shall provide clear visual feedback.

**Specifications**:
- Console messages for all major operations
- Visual indicators for cluster assignments
- Clear distinction between original groups and cluster assignments
- Intuitive interaction model (click to scatter, drag to move)

**Priority**: High

### 4.3 Data Quality (NFR-003)
**Requirement**: Data generation shall be reproducible.

**Specifications**:
- Use fixed random seed (42) for reproducibility
- Consistent overlap percentage across runs
- Predictable cluster center positions

**Priority**: Medium

---

## 5. User Interface Requirements

### 5.1 Plot Layout (UI-001)
**Specifications**:
- Figure size: 10x8 inches
- Single main axes for data display
- Bottom margin: 0.15 (to accommodate button)
- Grid: Enabled with alpha=0.3

### 5.2 Interactive Elements (UI-002)
**Specifications**:
- Button:
  - Position: [0.4, 0.02, 0.2, 0.05]
  - Label: "Run K-means"
  - Always enabled
- Mouse interactions:
  - Left-click near centroid without drag: Scatter cluster
  - Left-click and drag centroid: Move cluster
  - Click threshold: 0.5 units
  - Drag threshold: 0.1 units

### 5.3 Visual Elements (UI-003)
**Specifications**:
- Point size: 40 (clustered), 20 (original groups)
- Point transparency: 0.6 (clustered), 0.4 (original)
- Edge line width: 2 pixels (clustered points)
- Centroid marker: 'X', size 200, black with white edges
- Color scheme:
  - Groups: Red, Blue, Green
  - Clusters: Purple, Orange, Cyan (edges only)
  - Centroids: Black with white outline

---

## 6. Output Requirements

### 6.1 File Output (OUT-001)
**Requirement**: The system shall save visualization to file.

**Specifications**:
- Filename: `overlapping_groups.png`
- Resolution: 300 DPI
- Format: PNG
- Save on initial display (before user interaction)

### 6.2 Console Output (OUT-002)
**Requirement**: The system shall provide console feedback.

**Specifications**:
- Data generation statistics (group sizes, ranges)
- K-means iteration progress
- Convergence information
- User interaction confirmations (scatter, drag)
- File save confirmation

---

## 7. Constraints and Assumptions

### 7.1 Constraints
- Python 3.x environment required
- NumPy and Matplotlib must be installed
- Single-threaded execution
- 2D data only (no 3D support)

### 7.2 Assumptions
- User has basic understanding of K-means clustering
- Mouse/pointing device available for interaction
- Display supports color visualization
- Sufficient memory for 6,000+ data points

---

## 8. Future Enhancements (Out of Scope)

The following features are not included in the current requirements but may be considered for future versions:

- Variable number of clusters (K parameter adjustment)
- Variable number of points per group
- Different overlap percentages
- Alternative clustering algorithms
- 3D visualization support
- Export cluster assignments to file
- Animation of K-means convergence
- Undo/redo functionality

---

## 9. Acceptance Criteria

The system shall be considered complete when:

1. ✓ Three groups of 2,000 points each are generated with 20% overlap
2. ✓ Initial visualization displays all three groups with distinct colors
3. ✓ "Run K-means" button executes clustering algorithm
4. ✓ K-means converges based on centroid/label stability
5. ✓ Clustering results show original colors with cluster edge colors
6. ✓ Clicking near centroid scatters that cluster's points
7. ✓ Scattered points lose their edge color circles
8. ✓ Dragging centroid moves entire cluster
9. ✓ Re-running K-means clears previous results
10. ✓ Console provides informative feedback
11. ✓ Visualization saves to PNG file

---

## 10. Traceability Matrix

| Requirement ID | Description | Source Prompts | Priority |
|---------------|-------------|----------------|----------|
| FR-001 | Data Generation | 1 | High |
| FR-002 | Visualization | 1, 3, 4 | High |
| FR-003 | K-means Clustering | 5, 10 | High |
| FR-004.1 | Run K-means Button | 5, 12 | High |
| FR-004.2 | Click to Scatter | 6, 7, 9, 13 | Medium |
| FR-004.3 | Drag and Drop | 11 | Medium |
| NFR-001 | Performance | Implied | Medium |
| NFR-002 | Usability | Multiple | High |
| NFR-003 | Data Quality | 1 | Medium |

---

**End of Requirements Document**
