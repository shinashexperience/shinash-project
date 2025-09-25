import numpy as np

def calculate(numbers):
    if len(numbers) != 9:
        raise ValueError("List must contain nine numbers.")
    
    matrix = np.array(numbers).reshape(3, 3)
    
    calculations = {
        'mean': [
            np.mean(matrix, axis=0).tolist(),
            np.mean(matrix, axis=1).tolist(),
            np.mean(matrix).tolist()
        ],
        'variance': [
            np.var(matrix, axis=0).tolist(),
            np.var(matrix, axis=1).tolist(),
            np.var(matrix).tolist()
        ],
        'standard deviation': [
            np.std(matrix, axis=0).tolist(),
            np.std(matrix, axis=1).tolist(),
            np.std(matrix).tolist()
        ],
        'max': [
            np.max(matrix, axis=0).tolist(),
            np.max(matrix, axis=1).tolist(),
            np.max(matrix).tolist()
        ],
        'min': [
            np.min(matrix, axis=0).tolist(),
            np.min(matrix, axis=1).tolist(),
            np.min(matrix).tolist()
        ],
        'sum': [
            np.sum(matrix, axis=0).tolist(),
            np.sum(matrix, axis=1).tolist(),
            np.sum(matrix).tolist()
        ]
    }
    
    return calculations

# Test
if __name__ == "__main__":
    print(calculate([0,1,2,3,4,5,6,7,8]))