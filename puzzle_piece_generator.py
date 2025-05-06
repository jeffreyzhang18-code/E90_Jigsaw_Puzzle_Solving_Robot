import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle

def extract_image(image, center, angle, width, height, points, edges, contour):
    # Extract the given image from the box and rotate it to standard xy-axis
    shape = (image.shape[1], image.shape[0]) 
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    x = max(0, int(center[0] - width/2))
    y = max(0, int(center[1] - height/2))
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)

    if x >= x2 or y >= y2:
        return None, []  
    transformed_points = []
    for point in points:
        rotated_point = np.dot(matrix, np.array([point[0][0], point[0][1], 1.0]))
        
        rotated_point = rotated_point[:2]
        transformed_x = rotated_point[0] - x
        transformed_y = rotated_point[1] - y
        
        transformed_points.append((transformed_x, transformed_y))

    rotated_point = np.dot(matrix, np.array([center[0], center[1], 1.0]))
    rotated_point = rotated_point[:2]
    transformed_x = rotated_point[0] - x
    transformed_y = rotated_point[1] - y 
    transformed_center = (transformed_x, transformed_y)
    transformed_edges = []
    for edge in edges:
        transformed_edge = []
        for point in edge:
            rotated_point = np.dot(matrix, np.array([point[0], point[1], 1.0]))
            rotated_point = rotated_point[:2]
            transformed_x = rotated_point[0] - x 
            transformed_y = rotated_point[1] - y
            transformed_edge.append((transformed_x, transformed_y))
        transformed_edges.append(transformed_edge)
    
    contour_array = contour.squeeze()
    transformed_contour = []
    for point in contour_array:
        rotated_point = np.dot(matrix, np.array([point[0], point[1], 1.0]))
        transformed_x = rotated_point[0] - x
        transformed_y = rotated_point[1] - y
        transformed_contour.append([[transformed_x, transformed_y]])
    transformed_contour = np.array(transformed_contour, dtype=np.float32)

    return image[y:y2, x:x2], transformed_points, transformed_center, transformed_edges, transformed_contour

def find_corners(box, contour):
    # Find corners of puzzle by looking for pieces close to box corners and far from centroid
    puzzle_corners = []
    m = cv2.moments(contour)
    area = m['m00']
    centroid = np.array([m['m10'], m['m01']]) / area

    for i, corner in enumerate(box):
        min_weighted_dist = float('inf')
        closest_point = None
        closest_index = -1

        for j, pt in enumerate(contour):
            point = pt[0]
            centroid_diff = np.abs(point - centroid)
            centroid_dist = np.min(centroid_diff)  
            corner_dist = np.linalg.norm(point - corner)
            weighted_dist = corner_dist - 1.25 * centroid_dist

            if weighted_dist < min_weighted_dist:
                min_weighted_dist = weighted_dist
                closest_point = point
                closest_index = j

        puzzle_corners.append([closest_point, closest_index])
        puzzle_corners = sorted(puzzle_corners, key=lambda c: c[1])

    return puzzle_corners

def partition_edges(corners, contour):
    # Partition edges based on corners
    indices = [c[1] for c in corners]
    edges = []
    for i in range(4):
        start_idx = indices[i]
        end_idx = indices[(i + 1) % 4]

        if start_idx < end_idx:
            edge = contour[start_idx:end_idx + 1]
        else:
            edge = np.vstack((contour[start_idx:], contour[:end_idx]))

        edges.append(edge)

    return edges

def contour_arc_points(contour, x):
    # Find x equally spaced points along contour
    arc_length = cv2.arcLength(contour, closed=False)
    arc_interval = arc_length / (x + 1)
    contour = contour[:, 0, :].astype(np.float32)
    points = []
    dist_trav = 0
    current_target = arc_interval
    y = 0
    # points.append(tuple(map(int, contour[0])))
    while len(points) < x and y < len(contour) - 1:
        p1 = contour[y]
        p2 = contour[y + 1]
        seg_len = np.linalg.norm(p2 - p1)

        if dist_trav + seg_len >= current_target:
            frac = (current_target - dist_trav) / seg_len
            interp_point = (1 - frac) * p1 + frac * p2
            points.append(tuple(map(int, interp_point))) 
            current_target+=arc_interval
        dist_trav += seg_len
        y += 1
    # points.append(tuple(map(int, contour[-1])))
    return points

def find_curvature(points):
    # find curvature of given set of points
    n = len(points)
    curvature = []

    for i in range(n-2):
        p0 = np.array(points[i])
        p1 = np.array(points[i + 1])
        p2 = np.array(points[(i + 2) % n])
        v1 = p1 - p0
        v2 = p2 - p1
        cross_product = np.cross(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            curv = 0
        else:
            curv = cross_product / norm_product
        # print(curv)
        curvature.append(curv)
    return curvature

def classify_edge(edge, threshold = 0.1):
    # Classify edges as straight, curves in, curves out by curvature
    contour_points = contour_arc_points(edge, 4)
    contour_points = contour_points
    curvature = find_curvature(contour_points)
    average_curvature = sum(curvature)/len(curvature)
    if average_curvature < threshold and average_curvature > -threshold:
        return 'straight'
    elif average_curvature > threshold:
        return 'curves in'
    else:
        return 'curves out'

def matching_edges_distance(list1, list2, length_threshold=50):
    # find matches based on mean least squared 
    candidate_matches = []

    def compute_dists(points):
        return [np.linalg.norm(np.array(points[i]) - np.array(points[i + 1])) for i in range(len(points) - 1)]

    def dist_mse(dist1, dist2):
        length = min(len(dist1), len(dist2))
        return sum((dist1[i] - dist2[i]) ** 2 for i in range(length)) / length

    def compute_total_dist(points):
        return np.linalg.norm(np.array(points[0]) - np.array(points[-1]))
    for edge_out in list1:
        idx1_contour, idx1_edge, edge_out_points = edge_out
        dists_out = compute_dists(edge_out_points)

        for edge_in in list2:
            idx2_contour, idx2_edge, edge_in_points = edge_in

            if idx1_contour == idx2_contour:
                continue

            if abs(compute_total_dist(edge_out_points) - compute_total_dist(edge_in_points)) > length_threshold:
                continue

            dists_in_rev = compute_dists(list(reversed(edge_in_points)))

            score_rev = dist_mse(dists_out, dists_in_rev)

            candidate_matches.append((
                (idx1_contour, idx1_edge),
                (idx2_contour, idx2_edge),
                score_rev
            ))
    candidate_matches.sort(key=lambda x: x[2])
    matched_pairs = []
    used_out_edges = set()
    used_in_edges = set()

    for (out_id, in_id, score) in candidate_matches:
        if out_id not in used_out_edges and in_id not in used_in_edges:
            matched_pairs.append((out_id, in_id))
            used_out_edges.add(out_id)
            used_in_edges.add(in_id)
            print(f"Contour {out_id[0]} Edge {out_id[1]} ↔ Contour {in_id[0]} Edge {in_id[1]} (score: {score:.4f})")

    matched_pairs.sort(key=lambda x: x[0][0])
    
    return matched_pairs

def extract_contours(image_path, min_area, max_area):
    # extract individual contours and finds matches
    image = cv2.imread(image_path)
    structuring_element = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8)
    morph_iter = 8
    purple_mask = cv2.inRange(image, (160,100,140), (200, 140, 180))
    white_mask = cv2.inRange(image, (100, 0, 100), (255, 255, 255))
    combined_mask = cv2.bitwise_or(purple_mask, white_mask)
    combined_mask = cv2.dilate(combined_mask, structuring_element, iterations=morph_iter)
    combined_mask = cv2.erode(combined_mask, structuring_element, iterations=morph_iter)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if min_area < cv2.contourArea(contour) < max_area]
    mask = np.zeros_like(image)
    curves_in = []
    curves_out = []
    straight = []
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=10)
    extracted_shapes = cv2.bitwise_and(image, mask)
    piece_data = {}
    
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        corners = find_corners(box, contour)
        for point in corners:
            cv2.circle(extracted_shapes, point[0], 20, (0, 0, 255), 10, cv2.FILLED)
        edges = partition_edges(corners, contour)
        edge_list = []
        straight_edge_ct = 0
        cv2.putText(extracted_shapes, str(i), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA) 
        for j, edge in enumerate(edges):
            txt = classify_edge(edge)
            edge_idx = str(j)
            display_txt = txt + ' ' + edge_idx
            contour_points = contour_arc_points(edge, 8)
            center_x = sum(p[0] for p in contour_points) // len(contour_points)
            center_y = sum(p[1] for p in contour_points) // len(contour_points)
            # cv2.putText(extracted_shapes, display_txt, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
            if txt == 'straight':
                straight.append([i,j,contour_points])
                straight_edge_ct += 1
            elif txt == 'curves in':
                curves_in.append([i,j,contour_points])
            elif txt == 'curves out':
                curves_out.append([i,j,contour_points])
            for point in contour_points:
                 cv2.circle(extracted_shapes, point, 20, (255, 0, 0), 10, cv2.FILLED)
            edge_list.append(contour_points)
        if straight_edge_ct == 2:
            piece_type = 'corner'
        elif straight_edge_ct == 1:
            piece_type = 'edge'
        else:
            piece_type = 'center'
        extracted_image, corners, center, edge_list, contour = extract_image(extracted_shapes, center, angle, int(width), int(height), corners, edge_list, contour)
        cv2.imwrite(f"puzzle_1_flipped/pieces/piece_{i}_peaks.jpg", extracted_image)
        piece_data[i] = {
        "center": center,
        "contour": contour,
        "corners": corners,
        "edges": edge_list,
        "type": piece_type
        }
    print(f"Curves in: {len(curves_in)}, Curves out: {len(curves_out)}")
    for i in range(len(piece_data)):
        print(piece_data[i]["contour"])
    matched_edges = matching_edges_distance(curves_in,curves_out)
    with open("puzzle_1_flipped/piece_matches.txt", "w") as f:
        for i in range(len(matched_edges)):
            f.write(str(matched_edges[i][0][0]) + " ")
            f.write(str(matched_edges[i][0][1]) + " ")
            f.write(str(matched_edges[i][1][0]) + " ")
            f.write(str(matched_edges[i][1][1]) + "\n")
    with open("puzzle_1_flipped/piece_data.pkl", 'wb') as f:
        pickle.dump(piece_data, f)
def main():
    image_path = "puzzle_1_flipped/puzzle_1_combined.jpg"  
    min_area = 100000
    max_area = 100000000000000000

    start = timeit.default_timer()
    extract_contours(image_path, min_area, max_area)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
if __name__ == '__main__':
    main()