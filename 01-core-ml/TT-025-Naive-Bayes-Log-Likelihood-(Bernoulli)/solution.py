import math

def naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0):
    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")
    print(f"X_test: {X_test}")
    print(f"alpha: {alpha}")

    def ensure_2d(X):
        if X is None: return []
        if isinstance(X, (int, float)): return [[int(X)]]
        if isinstance(X, (list, tuple)):
            if len(X) == 0: return []
            if isinstance(X[0], (int, float)):
                return [[int(v) for v in X]]
            return [[int(v) for v in row] for row in X]
        return []

    def ensure_1d(y):
        if y is None: return []
        if isinstance(y, (int, float)): return [int(y)]
        if isinstance(y, (list, tuple)):
            if len(y) == 0: return []
            if isinstance(y[0], (list, tuple)):
                return [int(row[0]) for row in y]
            return [int(v) for v in y]
        return []

    X_train = ensure_2d(X_train)
    X_test  = ensure_2d(X_test)
    y_train = ensure_1d(y_train)

    print(f"After normalization:")
    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")
    print(f"X_test: {X_test}")

    if not X_train or not y_train or not X_test:
        return []

    n_samples  = len(X_train)
    n_features = len(X_train[0])

    classes     = sorted(set(y_train))
    class_stats = {}

    for c in classes:
        X_c   = [X_train[i] for i in range(n_samples) if y_train[i] == c]
        n_c   = len(X_c)
        prior = n_c / float(n_samples)

        count_ones = [0] * n_features
        for row in X_c:
            for j in range(n_features):
                if row[j] == 1:
                    count_ones[j] += 1

        probs = [(count_ones[j] + alpha) / (n_c + 2.0 * alpha) for j in range(n_features)]
        
        print(f"Class {c}: n_c={n_c}, prior={prior}, probs={probs}")
        
        class_stats[c] = {'prior': prior, 'probs': probs}

    predictions = []
    for x in X_test:
        print(f"Predicting for x={x}, type={type(x)}")
        class_scores = {}
        for c in classes:
            prior     = class_stats[c]['prior']
            probs     = class_stats[c]['probs']
            log_score = math.log(prior)

            for j in range(n_features):
                if j < len(x):
                    print(f"  j={j}, x[j]={x[j]}, type={type(x[j])}")
                    log_score += math.log(probs[j]) if x[j] == 1 else math.log(1.0 - probs[j])

            class_scores[c] = log_score

        predictions.append(int(max(class_scores, key=class_scores.get)))

    return predictions