import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle

def transform_func(points_a, points_b, center, corners, edges, contour=None, ax=None, label=None):
    def apply_xform(M, points):
        return np.matmul(points, M[:2, :2].T) + M[:2, 2]

    def get_canonical_xform(points):
        points = np.array(points, dtype=np.float32)
        midpoint = np.mean(points, axis=0)
        x_vector = points[-1] - points[0]
        x_vector /= np.linalg.norm(x_vector)
        u, v = x_vector
        M_first = np.array([[1, 0, -midpoint[0]],
                            [0, 1, -midpoint[1]],
                            [0, 0, 1]])
        M_second = np.array([[u, v, 0],
                             [-v, u, 0],
                             [0, 0, 1]])
        return np.matmul(M_second, M_first)

    canonical_from_A = get_canonical_xform(points_a)
    canonical_from_B = get_canonical_xform(points_b)
    A_from_B = np.matmul(np.linalg.inv(canonical_from_A), canonical_from_B)

    center_transformed = apply_xform(A_from_B, np.array([center]))[0]
    corners_transformed = apply_xform(A_from_B, np.array(corners))
    edges_transformed = [apply_xform(A_from_B, np.array(edge, dtype=np.float32).reshape(-1, 2)) for edge in edges]

    transformed_contour = None
    if contour is not None:
        contour_array = np.array(contour, dtype=np.float32).reshape(-1, 2)
        transformed_contour = apply_xform(A_from_B, contour_array)

        if ax is not None:
            plot_contour = transformed_contour.copy()
            plot_corners = corners_transformed.copy()
            plot_center = center_transformed.copy()
            plot_contour[:, 1] *= -1
            plot_corners[:, 1] *= -1
            plot_center[1] *= -1
            ax.plot(plot_contour[:, 0], plot_contour[:, 1], label=f'Contour {label}')
            ax.plot(plot_corners[:, 0], plot_corners[:, 1], 'ro')
            ax.text(plot_center[0], plot_center[1], str(label), fontsize=5, color='black')

    return center_transformed, corners_transformed, edges_transformed, transformed_contour

def main():
    matches_path = "puzzle_1_flipped/piece_matches.txt"
    start = timeit.default_timer()

    matches = np.loadtxt(matches_path, dtype=int)
    with open("puzzle_1_flipped/piece_data.pkl", 'rb') as f:
        piece_data = pickle.load(f)

    corner_count = sum(1 for p in piece_data.values() if p["type"] == "corner")
    edge_count = sum(1 for p in piece_data.values() if p["type"] == "edge")
    center_count = sum(1 for p in piece_data.values() if p["type"] == "center")
    print(f"Corner: {corner_count}, Edge: {edge_count}, Center: {center_count}")

    placed = set()
    fig, ax = plt.subplots()

    for match in matches:
        if piece_data[match[0]]["type"] == "corner":
            first_idx = match[0]
            break

    piece_info = piece_data[first_idx]
    center, corners, edges, contour = transform_func(
        [[-300, 0], [300, 0]],
        (piece_info["corners"][0], piece_info["corners"][1]),
        piece_info["center"],
        piece_info["corners"],
        piece_info["edges"],
        piece_info["contour"],
        ax,
        first_idx
    )
    piece_info["center"] = center
    piece_info["corners"] = corners
    piece_info["edges"] = edges
    piece_info["contour"] = contour
    placed.add(first_idx)

    while len(placed) < 36:
        placed_at_step = set()
        added = False
        for unplaced_idx in piece_data:
            if unplaced_idx in placed:
                continue

            candidate_transforms = []
            for match in matches:
                if match[0] == unplaced_idx and match[2] in placed:
                    placed_idx = match[2]
                    placed_edge_idx = match[3]
                    unplaced_edge_idx = match[1]
                elif match[2] == unplaced_idx and match[0] in placed:
                    placed_idx = match[0]
                    placed_edge_idx = match[1]
                    unplaced_edge_idx = match[3]
                else:
                    continue

                center, corners, edges, transformed_contour = transform_func(
                    piece_data[placed_idx]["edges"][placed_edge_idx],
                    piece_data[unplaced_idx]["edges"][unplaced_edge_idx][::-1],
                    piece_data[unplaced_idx]["center"],
                    piece_data[unplaced_idx]["corners"],
                    piece_data[unplaced_idx]["edges"],
                    piece_data[unplaced_idx]["contour"],
                    None,  
                    None
                )
                candidate_transforms.append((center, corners, edges, transformed_contour))

            if candidate_transforms:
                centers = np.array([c[0] for c in candidate_transforms])
                corners = np.array([c[1] for c in candidate_transforms])
                edges_list = [c[2] for c in candidate_transforms]
                contours = [c[3] for c in candidate_transforms]

                avg_center = np.mean(centers, axis=0)
                avg_corners = np.mean(corners, axis=0)
                avg_edges = []
                for i in range(len(edges_list[0])):
                    all_i = np.array([e[i] for e in edges_list])
                    avg_edges.append(np.mean(all_i, axis=0))

                if all(c is not None for c in contours):
                    contours_array = np.array(contours)
                    avg_contour = np.mean(contours_array, axis=0)
                else:
                    avg_contour = None

                piece_data[unplaced_idx]["center"] = avg_center
                piece_data[unplaced_idx]["corners"] = avg_corners
                piece_data[unplaced_idx]["edges"] = avg_edges
                piece_data[unplaced_idx]["contour"] = avg_contour
                
                if avg_contour is not None:
                    plot_contour = avg_contour.copy()
                    plot_contour[:, 1] *= -1
                    ax.plot(plot_contour[:, 0], plot_contour[:, 1], label=f'Center {unplaced_idx}')
                plot_corners = avg_corners.copy()
                plot_corners[:, 1] *= -1
                ax.plot(plot_corners[:, 0], plot_corners[:, 1], 'co')
                plot_center = avg_center.copy()
                plot_center[1] *= -1
                ax.text(plot_center[0], plot_center[1], str(unplaced_idx), fontsize=5, color='black')

                placed_at_step.add(unplaced_idx)
                added = True
                print(f"Averaged and placed center piece {unplaced_idx}")

        for piece in placed_at_step:
            placed.add(piece)
        if not added:
            print("No more center pieces could be placed with averaging.")
            break

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
