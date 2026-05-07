import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test,  dtype=np.float64)

    classes = np.unique(y_train)
    n_train = X_train.shape[0]

    log_theta, log_one_minus_theta, log_priors = [], [], []

    for c in classes:
        mask = (y_train == c)
        n_c  = int(mask.sum())
        theta = (X_train[mask].sum(axis=0) + alpha) / (n_c + 2.0 * alpha)
        log_theta.append(np.log(theta))
        log_one_minus_theta.append(np.log(1.0 - theta))
        log_priors.append(np.log(n_c / n_train))

    log_theta             = np.array(log_theta)
    log_one_minus_theta   = np.array(log_one_minus_theta)
    log_priors            = np.array(log_priors)

    return X_test @ log_theta.T + (1.0 - X_test) @ log_one_minus_theta.T + log_priors