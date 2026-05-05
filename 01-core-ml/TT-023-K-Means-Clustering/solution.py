import numpy as np

def distance(x1, y1):
    return np.sum((np.array(x1) - np.array(y1))**2)

def k_means_assignment(points, centroids):
    labels = []
    
    for j in range(len(points)):
        dist = []
        for centroid in centroids:
            dis = distance(points[j], centroid)
            dist.append(dis)
        
        nearest = int(np.argmin(dist))
        labels.append(nearest)

    # Returns a flat list of integers
    return labels
