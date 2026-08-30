from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

N_SAMPLES = 20_000
N_CLIENTS = 5

X, y = make_classification(n_samples=N_SAMPLES, n_features=4, n_classes=2,n_informative=4, n_redundant=0, random_state=42)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # X_test and y_test are the test data that will be used by server to evaluate the final model
X, X_base, y, y_base = train_test_split(X, y, test_size=0.5, random_state=42) # X_base and y_base are the baseline data that will be used by server to train the initial model

SLICE_SIZE = len(X) // N_CLIENTS

baseline_data_clients = [
    (X[i : i + SLICE_SIZE], y[i : i + SLICE_SIZE])
    for i in range(0, N_CLIENTS * SLICE_SIZE, SLICE_SIZE)
]

(
    baseline_data_client_1,
    baseline_data_client_2,
    baseline_data_client_3,
    baseline_data_client_4,
    baseline_data_client_5,
) = baseline_data_clients
