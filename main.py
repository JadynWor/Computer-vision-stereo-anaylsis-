import os
import cv2 as cv
import numpy as np

left_image = cv.imread("/Users/jadynworthington/Computer-vision-stereo-anaylsis-/venus_o_d.jpg")
right_image = cv.imread("/Users/jadynworthington/Computer-vision-stereo-anaylsis-/barn1_o_d.jpg")



def average_neighborhood(disparity):
    # Define the size of the neighborhood window for averaging (you can adjust this value)
    neighborhood_size = 3

    # Iterate through the disparity map
    for y in range(disparity.shape[0]):
        for x in range(disparity.shape[1]):
            if disparity[y, x] == 0:  # If a gap is detected
                # Define the region of interest (neighborhood window)
                y_start = max(0, y - neighborhood_size // 2)
                y_end = min(disparity.shape[0], y + neighborhood_size // 2 + 1)
                x_start = max(0, x - neighborhood_size // 2)
                x_end = min(disparity.shape[1], x + neighborhood_size // 2 + 1)

                # Collect valid disparity values from the neighborhood
                valid_values = [disparity[i, j] for i in range(y_start, y_end) for j in range(x_start, x_end) if disparity[i, j] != 0]

                if valid_values:  # If there are valid values in the neighborhood
                    # Fill the gap by computing the average of valid values
                    disparity[y, x] = sum(valid_values) / len(valid_values)
                else:
                    # If insufficient valid values, consider expanding the neighborhood size or use a different interpolation method
                    pass  # Handling case with insufficient valid values

    return disparity
   
def feature_based(left_image, right_image, DISTANCE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    # Apply Harris Corner Detection
    corners_left = cv.cornerHarris(cv.cvtColor(left_image, cv.COLOR_BGR2GRAY), 2, 3, 0.04)
    corners_right = cv.cornerHarris(cv.cvtColor(right_image, cv.COLOR_BGR2GRAY), 2, 3, 0.04)

    # Extract coordinates of detected corners
    corners_left = np.argwhere(corners_left > 0.01 * corners_left.max())
    corners_right = np.argwhere(corners_right > 0.01 * corners_right.max())

    # Initialize disparity map
    disparity = np.zeros_like(left_image)

    # Iterate through corners for feature matching
    for corner in corners_left:
        x, y = corner[1], corner[0]
        template = left_image[y - TEMPLATE_SIZE_Y // 2:y + TEMPLATE_SIZE_Y // 2,
                              x - TEMPLATE_SIZE_X // 2:x + TEMPLATE_SIZE_X // 2]

        best_match_x = -1
        best_match_score = float('inf') if DISTANCE in ['SAD', 'SSD'] else -1

        for corner_r in corners_right:
            x_r, y_r = corner_r[1], corner_r[0]
            if abs(y - y_r) > TEMPLATE_SIZE_Y // 2 or abs(x - x_r) > TEMPLATE_SIZE_X // 2:
                continue

            window = right_image[y_r - TEMPLATE_SIZE_Y // 2:y_r + TEMPLATE_SIZE_Y // 2,
                                 x_r - TEMPLATE_SIZE_X // 2:x_r + TEMPLATE_SIZE_X // 2]

            # Matching score computation
            if DISTANCE == 'SAD':
                score = np.sum(np.abs(template.astype(np.float32) - window.astype(np.float32)))
            elif DISTANCE == 'SSD':
                score = np.sum((template.astype(np.float32) - window.astype(np.float32)) ** 2)
            elif DISTANCE == 'NCC':
                # Similar NCC calculation as in region-based method

            # Update best match
                if (DISTANCE in ['SAD', 'SSD'] and score < best_match_score) or (DISTANCE == 'NCC' and score > best_match_score):
                    best_match_score = score
                    best_match_x = x_r

        # Assign disparity value for the best match
        disparity[y - TEMPLATE_SIZE_Y // 2:y + TEMPLATE_SIZE_Y // 2,
                 x - TEMPLATE_SIZE_X // 2:x + TEMPLATE_SIZE_X // 2] = np.abs(x - best_match_x)

    return disparity

def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
    # Initialize an empty disparity map
    disparity = np.zeros_like(left_image)

    # Iterate through the image pixels for template matching
    for y in range(0, left_image.shape[0] - TEMPLATE_SIZE_Y + 1):
        for x in range(0, left_image.shape[1] - TEMPLATE_SIZE_X + 1):
            template = left_image[y:y + TEMPLATE_SIZE_Y, x:x + TEMPLATE_SIZE_X]

            best_match_x = -1
            best_match_score = float('inf') if DISTANCE in ['SAD', 'SSD'] else -1  # Initializing best match score

            for offset in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
                if x + offset < 0 or x + offset + TEMPLATE_SIZE_X > right_image.shape[1]:
                    continue

                window = right_image[y:y + TEMPLATE_SIZE_Y, x + offset:x + offset + TEMPLATE_SIZE_X]

                if DISTANCE == 'SAD':
                    score = np.sum(np.abs(template.astype(np.float32) - window.astype(np.float32)))
                elif DISTANCE == 'SSD':
                    score = np.sum((template.astype(np.float32) - window.astype(np.float32)) ** 2)
                elif DISTANCE == 'NCC':
                    mean_template = np.mean(template.astype(np.float32))
                    std_template = np.std(template.astype(np.float32))
                    mean_window = np.mean(window.astype(np.float32))
                    std_window = np.std(window.astype(np.float32))

                    if std_template * std_window == 0:
                        score = -1  # Avoid division by zero
                    else:
                        score = np.sum(((template.astype(np.float32) - mean_template) * (window.astype(np.float32) - mean_window)) /
                                       (std_template * std_window))

                # Update best match
                if (DISTANCE in ['SAD', 'SSD'] and score < best_match_score) or (DISTANCE == 'NCC' and score > best_match_score):
                    best_match_score = score
                    best_match_x = x + offset

            # Assign disparity value for the best match
            disparity[y:y + TEMPLATE_SIZE_Y, x:x + TEMPLATE_SIZE_X] = np.abs(x - best_match_x)

    return disparity


method = input('Enter method [region, feature]: ')

if method == 'region' or method == 'feature':
    distance = input('Enter distance [SAD, SSD, NCC]: ')
    search_range = int(input('Enter search range (needs to be an integer): '))
    template_x_size = int(input('Enter template_x_size (needs to be an odd integer): '))
    template_y_size = int(input('Enter template_y_size (needs to be an odd integer): '))

    # Initialize disparity map (adjust this initialization according to your code)
    disparity = np.zeros_like(left_image)

#input('Enter distance [SAD, SSD, NCC]:'.format(i))
#input('Enter method [region, feature]:'.format(i))
#int(input('Enter search range (need to be integer):'.format(i)))
#int(input('Enter template_x_size (need to be odd integer):'.format(i)))
#int(input('Enter template_y_size (need to be odd integer):'.format(i)))


if method == 'region':
    disparity = region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y)
elif method == 'feature':
    disparity = feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y)

for i in range(2):
	disparity = average_neighborhood(disparity)

disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite('disparity.png', disparity)
