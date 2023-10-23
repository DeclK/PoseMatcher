import numpy as np
from .utils import get_keypoint_weight


class DTWForKeypoints:
    def __init__(self, keypoints1, keypoints2):
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
    
    def get_dtw_path(self):

        norm_kp1 = self.normalize_keypoints(self.keypoints1) #(frame_cnt 1, 17 , 2 )
        norm_kp2 = self.normalize_keypoints(self.keypoints2)

        kp_weight = get_keypoint_weight() #(17) , each stand for the weight of the joint
        oks, oks_unnorm = self.object_keypoint_similarity(norm_kp1,
                               norm_kp2, keypoint_weights=kp_weight) #(frame_cnt1 , frame_cnt2)
        print(f"OKS max {oks.max():.2f} min {oks.min():.2f}")

        # do the DTW, and return the path
        cost_matrix = 1 - oks
        dtw_dist, dtw_path = self.dynamic_time_warp(cost_matrix)

        return dtw_path, oks, oks_unnorm
        
    def normalize_keypoints(self, keypoints):
        centroid = keypoints.mean(axis=1)[:, None] #(frame_cnt , 2)
        max_distance = np.max(np.sqrt(np.sum((keypoints - centroid) ** 2, axis=2)),
                              axis=1) + 1e-6 #(frame_cnt)

        normalized_keypoints = (keypoints - centroid) / max_distance[:, None, None]
        return normalized_keypoints

    def keypoints_areas(self, keypoints):
        min_coords = np.min(keypoints, axis=1)
        max_coords = np.max(keypoints, axis=1)
        areas = np.prod(max_coords - min_coords, axis=1)
        return areas

    def get_best_match(self , keypoints1, keypoints2):  # keypoints1: (frame_cnt1, 17, 3) , keypoints2: (frame_cnt2, 17, 3)
        """Get the best match between two sets of keypoints.
        """
        new_keypoints = np.zeros((keypoints1.shape[0] , keypoints2.shape[0] , keypoints1.shape[1] , keypoints1.shape[2] ) )
        frame_cnt1 = keypoints1.shape[0] ; frame_cnt2 = keypoints2.shape[0]
        #trans_t = np.zeros((frame_cnt1,  frame_cnt2 , 3)) ; trans_R = np.zeros((frame_cnt1,  frame_cnt2 , 3 , 3))
        for i in range(frame_cnt1) :
            for j in range(frame_cnt2):
                source_points = keypoints1[i] ; target_points = keypoints2[j]
                source_mean =   np.mean(source_points , axis = 0) ; target_mean = np.mean(target_points , axis = 0)
                source_central = source_points - source_mean ; target_central = target_points - target_mean
                SVD_Matrix = target_central.T@source_central
                U , S , V = np.linalg.svd(SVD_Matrix)
                R = U@V
                if abs(np.linalg.det(R)  + 1) < 1e-2: 
                    V[: , 2] = -V[: , 2]
                    R = U@V
                # if iter_time == 0:
                #     print(R)
                assert(abs(np.linalg.det(R)  - 1) < 1e-2)
                t = target_mean.reshape(3,1) - R@source_mean.reshape(3,1)
                t = t.reshape(3)
                new_keypoints[i , j] = source_points@R.T + t
                
        return new_keypoints
        

    def object_keypoint_similarity(self, keypoints1,
                                keypoints2,
                                scale_constant=0.2,
                                keypoint_weights=None):
        """ Calculate the Object Keypoint Similarity (OKS) for multiple objects,
        and add weight to each keypoint. Here we choose to normalize the points
        using centroid and max distance instead of bounding box area.
        改进的点：在匹配中先加入ICP算法进行对齐，然后再进行匹配
        """
        # Compute squared distances between all pairs of keypoints
        
        new_keypoints = self.get_best_match(keypoints1 , keypoints2)
        
        sq_diff = np.sum((new_keypoints - keypoints2[None, ...]) ** 2, axis=-1)
        print("sq_diff shape is {}" .format(sq_diff.shape) , flush=True)
        
        oks = np.exp(-sq_diff / (2 * scale_constant ** 2))
        print("oks shape is {}" .format(oks.shape) , flush=True)
        oks_unnorm = oks.copy()
        
        if keypoint_weights is not None:
            oks = oks * keypoint_weights
            oks = np.sum(oks, axis=-1)
        else:
            oks = np.mean(oks, axis=-1)
        
        return oks, oks_unnorm

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