import numpy as np

def euclidean_distance(x1, y1):
    return np.sqrt(np.sum((np.array(x1) - np.array(y1))**2))

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # FIX 1: Explicitly set dtype=int for the empty case
    if len(X_test) == 0:
        return np.empty((0, k), dtype=int)

    all_results = []

    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])

        neighbors = []
        for j in range(k):
            if j < len(distances):
                neighbors.append(int(distances[j][1])) # Ensure index is int
            else:
                neighbors.append(-1) # This is an int
                
        all_results.append(neighbors)

    
    return np.array(all_results, dtype=int).reshape(len(X_test), k)

# Verification
result = knn_distance(X_train=[[0,0]], X_test=[], k=2)
print(f"Result: {result}")
print(f"Dtype: {result.dtype}") 
# Output: int64 or int32