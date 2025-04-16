from flask import Flask, request, jsonify, Blueprint
import numpy as np

tsvd_bp = Blueprint('tsvd', __name__)

# In-memory "database" for matrices
matrices_db = {}
next_id = 1

@tsvd_bp.route('/')
def home():
    return 'Matrix SVD API with optional truncation'

# POST
#  - Expects JSON with a 'matrix' key, containing a 2D list of numbers.
#  - Example: { "matrix": [[1,2],[3,4]] }
@tsvd_bp.route('/', methods=['POST'])
def create_matrix():
    global next_id
    
    # Parse JSON input
    data = request.get_json()
    matrix_data = data.get('matrix', None)

    # Validate input
    if matrix_data is None or not isinstance(matrix_data, list):
        return jsonify({'error': 'You must provide a "matrix" as a 2D list.'}), 400

    # Convert to a NumPy array
    try:
        np_matrix = np.array(matrix_data, dtype=float)
        # Validate it's 2D
        if np_matrix.ndim != 2:
            raise ValueError('Matrix must be 2D')
    except ValueError as e:
        return jsonify({'error': f'Invalid matrix data. {str(e)}'}), 400

    # Compute SVD: U, S, Vt
    U, s, Vt = np.linalg.svd(np_matrix, full_matrices=False)
    
    # Store in memory with a unique ID
    matrices_db[next_id] = {
        'matrix': np_matrix.tolist(),
        'U': U.tolist(),
        'S': s.tolist(),
        'Vt': Vt.tolist()
    }

    # Return the new ID
    response = {'id': next_id}
    next_id += 1
    return jsonify(response), 201


# GET /<id>?rank=<rank>
#  - Returns the original matrix plus U, S, and Vt
@tsvd_bp.route('/<int:matrix_id>', methods=['GET'])
def get_matrix_svd(matrix_id):
    matrix_entry = matrices_db.get(matrix_id)
    if not matrix_entry:
        return jsonify({'error': 'Matrix not found'}), 404

    # Parse rank from query parameter (e.g. ?rank=2)
    rank_str = request.args.get('rank', default='', type=str)
    
    if rank_str == '':
        # No rank provided, return full SVD
        return jsonify({
            'id': matrix_id,
            'matrix': matrix_entry['matrix'],
            'U': matrix_entry['U'],
            'S': matrix_entry['S'],
            'Vt': matrix_entry['Vt']
        }), 200

    # If rank is not provided or invalid, just return the full SVD
    if not rank_str.isdigit() or int(rank_str) <= 0:
        # rank not provided or invalid (not an positive integer)
        return jsonify({
            'error': 'Invalid rank parameter. Must be a positive integer.'
        }), 400
    
    # We have a valid integer rank
    rank_val = int(rank_str)

    # Convert stored lists back to NumPy arrays for slicing
    U = np.array(matrix_entry['U'])
    S = np.array(matrix_entry['S'])
    Vt = np.array(matrix_entry['Vt'])

    # The maximum rank we can use is limited by the length of S
    max_rank = len(S)

    # The actual rank we will use is the min of (rank_val, max_rank)
    r = min(rank_val, max_rank)

    # Truncate U, S, and Vt
    # U is (m x k), S is (k,), Vt is (k x n) if full_matrices=False
    U_trunc = U[:, :r]
    S_trunc = S[:r]
    Vt_trunc = Vt[:r, :]

    # Convert back to lists for JSON
    U_trunc_list = U_trunc.tolist()
    S_trunc_list = S_trunc.tolist()
    Vt_trunc_list = Vt_trunc.tolist()

    return jsonify({
        'id': matrix_id,
        'matrix': matrix_entry['matrix'],
        'U': U_trunc_list,
        'S': S_trunc_list,
        'Vt': Vt_trunc_list
    }), 200
