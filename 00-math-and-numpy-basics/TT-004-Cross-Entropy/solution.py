import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    if np.shape(y_pred) == np.shape(y_true):
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    elif len(np.shape(y_true)) == 1:
        correct_confidences = y_pred[np.arange(len(y_pred)), y_true]
        loss = -np.mean(np.log(correct_confidences))
        return loss

    else:
        raise ValueError("Incompatible shapes for y_true and y_pred.")


if __name__ == "__main__":
    y_true = [[1, 0, 0], [0, 1, 0]]
    y_pred = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
    print(cross_entropy_loss(y_true, y_pred))