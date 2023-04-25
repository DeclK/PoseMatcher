import numpy as np
from utils import get_keypoint_weight


class DTWForKeypoints:
    def __init__(self, keypoints1, keypoints2):
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
    
    def get_dtw_path(self):

        norm_kp1 = self.normalize_keypoints(self.keypoints1)
        norm_kp2 = self.normalize_keypoints(self.keypoints2)

        kp_weight = get_keypoint_weight()
        oks = self.object_keypoint_similarity(norm_kp1,
                                              norm_kp2,
                                              keypoint_weights=kp_weight)
        print(f"OKS max {oks.max():.2f} min {oks.min():.2f}")

        # do the DTW, and return the path
        cost_matrix = 1 - oks
        dtw_dist, dtw_path = self.dynamic_time_warp(cost_matrix)

        return dtw_path
        
    def normalize_keypoints(self, keypoints):
        centroid = keypoints.mean(axis=1)[:, None]
        max_distance = np.max(np.sqrt(np.sum((keypoints - centroid) ** 2, axis=2)),
                              axis=1) + 1e-6

        normalized_keypoints = (keypoints - centroid) / max_distance[:, None, None]
        return normalized_keypoints

    def keypoints_areas(self, keypoints):
        min_coords = np.min(keypoints, axis=1)
        max_coords = np.max(keypoints, axis=1)
        areas = np.prod(max_coords - min_coords, axis=1)
        return areas

    def object_keypoint_similarity(self, keypoints1,
                                keypoints2,
                                scale_constant=0.5,
                                keypoint_weights=None):
        """ Calculate the Object Keypoint Similarity (OKS) for multiple objects,
        and add weight to each keypoint. Here we choose to normalize the points
        using centroid and max distance instead of bounding box area.
        """
        # Compute squared distances between all pairs of keypoints
        sq_diff = np.sum((keypoints1[:, None] - keypoints2) ** 2, axis=-1)
        
        oks = np.exp(-sq_diff / (2 * scale_constant ** 2))
        
        if keypoint_weights is not None:
            oks = oks * keypoint_weights
            oks = np.sum(oks, axis=-1)
        else:
            oks = np.mean(oks, axis=-1)
        
        return oks

    def dynamic_time_warp(self, cost_matrix, R=1000):
        """Compute the Dynamic Time Warping distance and path between two time series.
        If the time series is too long, it will use the Sakoe-Chiba Band constraint,
        so time complexity is bounded at O(MR).
        """
        
        M = len(self.keypoints1)
        N = len(self.keypoints2)

        # Initialize the distance matrix with infinity
        D = np.full((M, N), np.inf)

        # Initialize the first row and column of the matrix
        D[0, 0] = cost_matrix[0, 0]
        for i in range(1, M):
            D[i, 0] = D[i - 1, 0] + cost_matrix[i, 0]

        for j in range(1, N):
            D[0, j] = D[0, j - 1] + cost_matrix[0, j]

        # Fill the remaining elements of the matrix within the
        # Sakoe-Chiba Band using dynamic programming
        for i in range(1, M):
            for j in range(max(1, i - R), min(N, i + R + 1)):
                cost = cost_matrix[i, j]
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        # Backtrack to find the optimal path
        path = [(M - 1, N - 1)]
        i, j = M - 1, N - 1
        while i > 0 or j > 0:
            min_idx = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
            path.append((i, j))
        path.reverse()

        return D[-1, -1], path

if __name__ == '__main__':

    from mmengine.fileio import load

    keypoints1, kp1_scores = load('tennis1.pkl')
    keypoints2, kp2_scores = load('tennis3.pkl')

    # Normalize the keypoints
    dtw = DTWForKeypoints(keypoints1, keypoints2)
    path = dtw.get_dtw_path()
    print(path)