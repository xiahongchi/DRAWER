def match(gt, pred):
    """
    Match prediction components to ground truth components with stricter pattern matching.
    
    Args:
        gt: Tuple of two lists (upper, lower) with (x,y) coordinates
        pred: Tuple of two lists (upper, lower) with (x,y) coordinates
        
    Returns:
        List of matching tuples
    """
    gt_upper, gt_lower = gt
    pred_upper, pred_lower = pred
    
    # Store original points
    original_gt_upper = gt_upper.copy()
    original_gt_lower = gt_lower.copy()
    original_pred_upper = pred_upper.copy()
    original_pred_lower = pred_lower.copy()
    
    # Normalize coordinates
    gt_upper_norm = normalize_coordinates(gt_upper)
    gt_lower_norm = normalize_coordinates(gt_lower)
    pred_upper_norm = normalize_coordinates(pred_upper)
    pred_lower_norm = normalize_coordinates(pred_lower)
    
    # Group points by x-coordinate
    gt_upper_groups = group_by_x(gt_upper_norm, original_gt_upper)
    gt_lower_groups = group_by_x(gt_lower_norm, original_gt_lower)
    pred_upper_groups = group_by_x(pred_upper_norm, original_pred_upper)
    pred_lower_groups = group_by_x(pred_lower_norm, original_pred_lower)
    
    # Find matches that preserve patterns
    upper_matches = find_pattern_preserving_matches(gt_upper_groups, pred_upper_groups, "upper")
    lower_matches = find_pattern_preserving_matches(gt_lower_groups, pred_lower_groups, "lower")
    
    return upper_matches + lower_matches

def normalize_coordinates(points):
    """Normalize x coordinates to (0,1) range."""
    if not points:
        return []
    
    x_values = [p[0] for p in points]
    min_x, max_x = min(x_values), max(x_values)
    
    if min_x == max_x:
        return [(0, p[1]) for p in points]
    
    normalized = []
    for x, y in points:
        norm_x = (x - min_x) / (max_x - min_x)
        normalized.append((norm_x, y))
    
    return normalized

def group_by_x(normalized_points, original_points):
    """Group points by their x-coordinate."""
    groups = {}
    for i, (norm_x, norm_y) in enumerate(normalized_points):
        # Round to handle floating point precision issues
        rounded_x = round(norm_x, 4)
        if rounded_x not in groups:
            groups[rounded_x] = []
        groups[rounded_x].append((i, norm_y, original_points[i]))
    
    return groups

def find_pattern_preserving_matches(gt_groups, pred_groups, location):
    """Find matches that preserve vertical patterns at each x-coordinate."""
    matches = []
    matched_gt_x = set()
    matched_pred_x = set()
    
    # Sort x-coordinates to process in order from left to right
    gt_x_values = sorted(gt_groups.keys())
    pred_x_values = sorted(pred_groups.keys())
    
    # Calculate distance between each possible pair of x-coordinate groups
    x_distances = []
    for gt_x in gt_x_values:
        for pred_x in pred_x_values:
            distance = abs(gt_x - pred_x)
            x_distances.append((distance, gt_x, pred_x))
    
    x_distances.sort()  # Sort by distance
    
    # Match groups that have the same pattern
    for _, gt_x, pred_x in x_distances:
        # Skip if either x is already matched
        if gt_x in matched_gt_x or pred_x in matched_pred_x:
            continue
        
        gt_points = gt_groups[gt_x]
        pred_points = pred_groups[pred_x]
        
        # Only match if the patterns have the same number of components
        if len(gt_points) == len(pred_points):
            # Sort by y-coordinate to ensure pattern matching
            gt_points.sort(key=lambda p: p[1])
            pred_points.sort(key=lambda p: p[1])
            
            # Check if patterns match
            pattern_matches = []
            for (gt_idx, gt_y, gt_orig), (pred_idx, pred_y, pred_orig) in zip(gt_points, pred_points):
                gt_orig_x, gt_orig_y = gt_orig
                pred_orig_x, pred_orig_y = pred_orig
                pattern_matches.append(
                    ((gt_orig_x, gt_orig_y, location), 
                     (pred_orig_x, pred_orig_y, location))
                )
            
            # If we found a matching pattern, add all matches
            matches.extend(pattern_matches)
            matched_gt_x.add(gt_x)
            matched_pred_x.add(pred_x)
    
    return matches

def evaluate_prediction(gt, pred):
    """Calculate evaluation metrics with stricter pattern matching."""
    gt_upper, gt_lower = gt
    pred_upper, pred_lower = pred
    
    total_predictions = len(pred_upper) + len(pred_lower)
    total_gt = len(gt_upper) + len(gt_lower)
    
    matches = match(gt, pred)
    accurate_predictions = len(matches)
    
    prediction_recall = accurate_predictions / total_gt if total_gt > 0 else 0
    prediction_accuracy = accurate_predictions / total_predictions if total_predictions > 0 else 0
    
    return {
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "accurate_predictions": accurate_predictions,
        "prediction_recall": prediction_recall,
        "prediction_accuracy": prediction_accuracy
    }

test_case = "uw_kitchen_3"

if test_case == "uw_kitchen":
    
    gt = ([(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1), (10,1)], 
        [(1,1), (2,1), (3,1), (4,1), (5,1), (5,2), (6,1), (6,2), (7,1), (8,1)])

    urdformer = ([(1,1), (2,1), (3,1)], 
                [(1,1), (2,1), (3,1), (4,1)])

    acdc = ([(1,1), (1,2), (2,1), (2,2), (3,1), (4,1), (5,1), (5,2), (6,1), (6,2), (7,1), (7,2), (8,1), (8,2), (9,1), (10,1)],
    [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (6,1)])

    ours = ([(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1)], [(1,1), (2,1), (3,1), (4,1), (5,1), (5,2), (6,1), (6,2), (7,1)])

elif test_case == "cs_kitchen":
    gt = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2)])
    urdformer = ([], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1)])
    acdc = ([(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2)], 
            [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2)])
    ours = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2)])
    
elif test_case == "cs_kitchen_n":
    gt = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)])
    urdformer = ([(1, 1), (2, 1), (3, 1), (4, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1), (6, 2)])
    acdc = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (5, 1)])
    ours = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (4, 2), (5, 1), (5, 2)])

elif test_case == "uw_kitchen_2":
    gt = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)], 
          [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (7, 1), (7, 2), (8, 1), (8, 2)])
    urdformer = ([(1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (5, 1), (5, 2)])
    acdc = ([(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (9, 1), (10, 1), (10, 2), (11, 1), (11, 2), (12, 1), (12, 2)], 
            [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (8, 2)])
    ours = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1), (7, 1), (7, 2), (8, 1), (8, 2)])

elif test_case == "uw_kitchen_3":
    gt = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)], [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (8, 1), (8, 2)])
    urdformer = ([(1, 1), (2, 1), (3, 1), (4, 1)], [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (5, 1), (6, 1), (6, 2)])
    acdc = ([(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (8, 2), (9, 1), (9, 2), (10, 1), (10, 2), (11, 1), (11, 2)], 
            [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (9, 1)])
    ours = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)], [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (8, 1), (8, 2)])

elif test_case == "cc_kitchen_p1":
    gt = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)])
    urdformer = ([(1, 1), (2, 1)], [(1, 1), (1, 2), (2, 1), (3, 1)])
    acdc = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1)])
    ours = ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2)])
    
elif test_case == "cc_kitchen_p2":
    gt = ([], [(1, 1), (1, 2), (2, 1), (3, 1)])
    urdformer = ([], [(1, 1), (1, 2)])
    acdc = ([], [(1, 1), (2, 1), (3, 1)])
    ours = ([], [(1, 1), (1, 2), (2, 1), (3, 1)])
    
for pred in [urdformer, acdc, ours]:
    # Get matches
    matches = match(gt, pred)
    # print("Matches:", matches)

    # Evaluate prediction
    metrics = evaluate_prediction(gt, pred)
    print("Metrics:", metrics)