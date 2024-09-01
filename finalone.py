import cv2 as cv
import numpy as np
import random
from tqdm.notebook import tqdm
from cornerdetection import gaussian_smoothing, gradient, r_score, extract_corners
#(1)
def process_image(image_path):
    img_float = np.float32(image_path)

    # Smooth the image
    smoothed_image = gaussian_smoothing(img_float, 1)

    # Compute gradients
    Gx, Gy = gradient(smoothed_image)

    # Compute the Harris corner response
    r = r_score(Gx, Gy, 5, 0.09)
    threshold = np.max(r) * 0.01

    # Extract corners 
    corners = extract_corners(r, threshold, max_corners=100000)
    corners_coords = []
    # Draw the corners 
    img_with_corners = cv.cvtColor(image_path, cv.COLOR_GRAY2BGR)
    for corner in corners:
        cv.circle(img_with_corners, (corner[1], corner[0]), 3, (0, 0, 255), -1)
#(2)
    # For patches extraction 
    img_with_rectangles = cv.cvtColor(image_path, cv.COLOR_GRAY2BGR)
    patch_size = 23
    patches = []
    half_patch_size = patch_size // 2

    for corner in corners:
        x, y = corner
        corners_coords.append((x, y))
        # Calculate the top-left and bottom-right corners of the patch
        x1, y1 = x - half_patch_size, y - half_patch_size
        x2, y2 = x + half_patch_size + 1, y + half_patch_size + 1

        # Check if the patch is within the image boundaries
        if x1 >= 0 and y1 >= 0 and x2 <= image_path.shape[1] and y2 <= image_path.shape[0]:
            patch = image_path[y1:y2, x1:x2]
            patches.append(patch)
    for corner in corners:
        x, y = corner
        top_left = (y - half_patch_size, x - half_patch_size)
        bottom_right = (y + half_patch_size, x + half_patch_size)
        cv.rectangle(img_with_rectangles, top_left, bottom_right, (0, 255, 0), 1)

    return patches, corners_coords,img_with_corners, img_with_rectangles

#(3)
def normalize_patches(patches):
   norm_patches = []
   for patch in patches:
        mean = np.mean(patch)
        std = np.std(patch)

        # Handle the case where standard deviation is zero
        if std != 0:
            norm_patch = (patch - mean) / std
        else:
            # If std is zero, the patch has constant values; it can be set to all zeros
            norm_patch = np.zeros_like(patch)

        norm_patches.append(norm_patch)
   return np.array(norm_patches)

#compute ncc score
def compute_ncc(norm_patches_src, norm_patches_dst):
    # Flatten the patches to 1D vectors
    flat_patches_src = norm_patches_src.reshape(norm_patches_src.shape[0], -1)
    flat_patches_dst = norm_patches_dst.reshape(norm_patches_dst.shape[0], -1)
    ncc_scores = np.dot(flat_patches_src, flat_patches_dst.T)
    return ncc_scores  

    

#find matches 
def find_bidirectional_matches(ncc_scores):
    # Forward Matching (Source to Destination)
    forward_matches = np.argmax(ncc_scores, axis=1)

    # Backward Matching (Destination to Source)
    backward_matches = np.argmax(ncc_scores, axis=0)

    # Bidirectional Matching
    matched_src, matched_dst = [], []
    for src_idx, dst_idx in enumerate(forward_matches):
        if backward_matches[dst_idx] == src_idx:
            matched_src.append(src_idx)
            matched_dst.append(dst_idx)

    return matched_src, matched_dst

def draw_matches(img_src, img_dst, src_corners, dst_corners, matched_src, matched_dst):
    h = max(img_src.shape[0], img_dst.shape[0])
    w = img_src.shape[1] + img_dst.shape[1]
    img_combined = np.zeros((h, w, 3), dtype=np.uint8)

    img_combined[:img_src.shape[0], :img_src.shape[1]] = cv.cvtColor(img_src, cv.COLOR_GRAY2BGR)
    img_combined[:img_dst.shape[0], img_src.shape[1]:] = cv.cvtColor(img_dst, cv.COLOR_GRAY2BGR)
    patch_size=23
    for src_idx, dst_idx in zip(matched_src, matched_dst):
        # Calculate the center of each patch (ensure coordinates are integers)
        
        # Debugging: Print corner data
        print("src_corners[src_idx]:", src_corners[src_idx])
        print("dst_corners[dst_idx]:", dst_corners[dst_idx])

        # Ensure that the coordinates are integers and in the correct format
        src_center = (int(src_corners[src_idx][1] + patch_size // 2), int(src_corners[src_idx][0] + patch_size // 2))
        dst_center = (int(dst_corners[dst_idx][1] + patch_size // 2) + img_src.shape[1], int(dst_corners[dst_idx][0] + patch_size // 2))
        src_center = (int(src_corners[src_idx][1] + patch_size // 2), int(src_corners[src_idx][0] + patch_size // 2))
        dst_center = (int(dst_corners[dst_idx][1] + patch_size // 2) + img_src.shape[1], int(dst_corners[dst_idx][0] + patch_size // 2))

        cv.line(img_combined, src_center, dst_center, (0, 255, 0), 1)  # Green line

    cv.imshow("Matched Patches", img_combined)
    cv.waitKey(0)
    cv.destroyAllWindows()
def create_matches_array(matched_src, matched_dst, src_corners, dst_corners):
    # Initialize an empty list for matches
    matches = []

    # For each pair of matched indices, get the corresponding coordinates
    for src_idx, dst_idx in zip(matched_src, matched_dst):
        x1, y1 = src_corners[src_idx]
        x2, y2 = dst_corners[dst_idx]
        matches.append([x1, y1, x2, y2])

    return np.array(matches)

# Assuming matched_src, matched_dst, src_corners, dst_corners are defined



def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H
def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)
def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors
#(5)
def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    return best_inliers, best_H


def draw_matches_inliers(img_src, img_dst, inliers):
    # Create a blank image with enough space to hold both images side by side
    h = max(img_src.shape[0], img_dst.shape[0])
    w = img_src.shape[1] + img_dst.shape[1]
    img_combined = np.zeros((h, w, 3), dtype=np.uint8)

    # Place the source and destination images side by side
    img_combined[:img_src.shape[0], :img_src.shape[1]] = cv.cvtColor(img_src, cv.COLOR_GRAY2BGR)
    img_combined[:img_dst.shape[0], img_src.shape[1]:] = cv.cvtColor(img_dst, cv.COLOR_GRAY2BGR)

    # Draw lines for inliers
    for match in inliers:
        x1, y1, x2, y2 = match
        src_point = (int(x1), int(y1))
        dst_point = (int(x2) + img_src.shape[1], int(y2))  # Offset x-coordinate

        cv.line(img_combined, src_point, dst_point, (0, 255, 0), 1)  # Green line

    # Display the image
    cv.imshow("Inlier Matches", img_combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

#stiching and backward warping together
def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


if __name__ == "__main__":
    src_image_path = "image1.jpg"
    dst_image_path = "image2.jpg"

    # Read the source and destination images
    src_image = cv.imread(src_image_path, cv.IMREAD_GRAYSCALE)
    dst_image = cv.imread(dst_image_path, cv.IMREAD_GRAYSCALE)
    src_image_colored = cv.imread(src_image_path)
    dst_image_colored = cv.imread(dst_image_path)
    # Ensure images are correctly loaded
    if src_image is None or dst_image is None:
        print("Error loading one or both images.")
        exit()
    src_patches,src_corners,src_img_corners,src_img_rect=process_image(src_image)
    dst_patches,dst_corners,dst_img_corners,dst_img_rect=process_image(dst_image)
    cv.imwrite("Src_corners.jpg",src_img_corners)
    
    cv.imwrite("Dst_corners.jpg",dst_img_corners)
    
    cv.imwrite("Src_patches.jpg",src_img_rect)
    
    cv.imwrite("Dst_patches.jpg",dst_img_rect)
    
    norm_patch_src=normalize_patches(src_patches)
    norm_patch_dst=normalize_patches(dst_patches)
    scores=compute_ncc(norm_patch_src,norm_patch_dst)
    
    matched_src, matched_dst = find_bidirectional_matches(scores)

# # Visualize the matches
    draw_matches(src_image, dst_image, src_corners, dst_corners, matched_src, matched_dst)
    matches = create_matches_array(matched_src, matched_dst, src_corners, dst_corners)
    inliers, H = ransac(matches, 0.5, 2000)

    draw_matches_inliers(src_image, dst_image, inliers)
    stitched_image = stitch_img(src_image_colored, dst_image_colored, H)

    # Show the stitched image
    cv.imwrite("stiching.jpg", stitched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
