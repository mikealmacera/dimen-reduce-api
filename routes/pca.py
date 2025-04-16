from flask import Flask, request, jsonify, Blueprint
import numpy as np

pca_bp = Blueprint('pca', __name__)

# In-memory storage: { id: { 'matrix': ..., 'mean': ..., 'components': ..., etc. } }
pca_db = {}
next_id = 1

@pca_bp.route('/')
def home():
    return 'PCA API with optional rank truncation'

# POST
# Body JSON format:
# {
#   "matrix": [[...], [...], ... ]  // shape (n_samples, n_features)
# }
@pca_bp.route('/', methods=['POST'])
def create_pca():
    global next_id

    data = request.get_json()
    matrix_data = data.get('matrix')

    # Validate input
    if matrix_data is None or not isinstance(matrix_data, list):
        return jsonify({'error': 'You must provide a "matrix" as a 2D list.'}), 400

    try:
        X = np.array(matrix_data, dtype=float)
        # Must be 2D for PCA (n_samples x n_features)
        if X.ndim != 2:
            raise ValueError('Matrix must be 2-dimensional')
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError('Need at least 2 samples for PCA')
    except ValueError as e:
        return jsonify({'error': f'Invalid matrix data: {str(e)}'}), 400

    # 1. Center the data by subtracting mean of each column
    mean_ = X.mean(axis=0)
    X_centered = X - mean_

    # 2. Compute SVD on the centered data
    #    X_centered = U * s * Vt
    #    PCA components = rows of Vt (or columns of V)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Compute explained variance and explained variance ratio
    #    Variance explained by each singular value s_i is (s_i^2) / (n_samples - 1)
    #    Total variance is the sum of variances across all original features in X_centered.
    #    A quick way is sum(s^2)/(n_samples - 1), but more precisely we can also compare
    #    with the sum of each column's variance if needed. For standard PCA on a centered X:
    #    total_variance = np.sum(np.var(X_centered, axis=0, ddof=1))  # ddof=1 for sample var
    #    explained_variance = (s**2) / (n_samples - 1)
    #    explained_variance_ratio = explained_variance / total_variance

    explained_variance = (s**2) / (n_samples - 1)
    total_variance = np.sum(np.var(X_centered, axis=0, ddof=1))
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else 0

    # Store all results in memory
    pca_db[next_id] = {
        'matrix': X.tolist(),
        'mean': mean_.tolist(),
        'components': Vt,  # shape: (n_components, n_features)
        'singular_values': s,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
    }

    response = {'id': next_id}
    next_id += 1
    return jsonify(response), 201

# GET /<id>?rank=N
# Returns the original matrix, plus PCA results:
#   mean, principal components, explained variance, explained variance ratio, etc.
# If rank=N is provided, results are truncated to the first N components.
@pca_bp.route('/<int:pca_id>', methods=['GET'])
def get_pca(pca_id):
    entry = pca_db.get(pca_id)
    if not entry:
        return jsonify({'error': 'PCA entry not found'}), 404

    rank_str = request.args.get('rank', default='', type=str)

    # Convert stored data back to NumPy
    components_full = entry['components']  # shape (min(n_samples, n_features), n_features)
    s_full = entry['singular_values']
    var_full = entry['explained_variance']
    ratio_full = entry['explained_variance_ratio']

    if rank_str == '':
        return jsonify({
            'id': pca_id,
            'matrix': entry['matrix'],
            'mean': entry['mean'],
            'components': components_full.tolist(),
            'singular_values': s_full.tolist(),
            'explained_variance': var_full.tolist(),
            'explained_variance_ratio': ratio_full.tolist()
        }), 200

    if not rank_str.isdigit() or int(rank_str) <= 0:
        # rank not provided or invalid (not a positive integer)
        return jsonify({
            'error': 'Invalid rank parameter. Must be a positive integer.'
        }), 400

    # We have a valid integer rank
    rank_val = int(rank_str)

    # Truncate to rank_val
    max_rank = len(s_full)
    r = min(rank_val, max_rank)

    components_trunc = components_full[:r, :]
    s_trunc = s_full[:r]
    var_trunc = var_full[:r]
    ratio_trunc = ratio_full[:r]

    return jsonify({
        'id': pca_id,
        'matrix': entry['matrix'],
        'mean': entry['mean'],
        'components': components_trunc.tolist(),
        'singular_values': s_trunc.tolist(),
        'explained_variance': var_trunc.tolist(),
        'explained_variance_ratio': ratio_trunc.tolist()
    }), 200