# coding:utf-8
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
"""
Matrix Factorizationの実体
ライブラリライクに使えるようにした
"""


class MyMFals:

    def __init__(self, aim_matrix, num_users, num_items, k=100, max_iteration=20, converge_threshold=0.0001):

        self.aim_lil = aim_matrix
        self.aim_csc = aim_matrix.tocsc()
        self.aim_csr = aim_matrix.tocsr()
        self.aim_transpose = self.aim_csc.transpose()
        self.k = k
        self.max_iteration = max_iteration
        self.converge_threshold = converge_threshold
        self.num_users = num_users
        self.num_items = num_items
        self.total_mean, self.user_means, self.item_means = self.find_means()
        self.user_matrix = None
        self.item_matrix = None

        # Subtract general, user-specific and item-specific bias so that aim-matrix show pure user-item compatibility
        for (row, col) in zip(*self.aim_csr.nonzero()):
            self.aim_csr[row, col] -= self.total_mean + self.user_means[row] + self.item_means[col]
        self.aim_csc = self.aim_csr.tocsc()
        self.aim_transpose = self.aim_csc.transpose()

    def find_means(self):
        total_mean = 0
        total_count = 0
        user_means = np.array([0.0] * self.num_users)
        item_means = np.array([0.0] * self.num_items)

        # Find mean-score of individual users and all users
        for (i, user_row) in enumerate(self.aim_csr):
            sum = user_row.sum()
            total_mean += sum
            num_score = user_row.count_nonzero()
            total_count += num_score
            if num_score > 0:
                user_means[i] = sum / num_score
            else:
                user_means[i] = 3.0

        total_mean /= total_count

        # Get bias of individual users respect to total-mean
        for i, user in enumerate(user_means):
            user_means[i] -= total_mean

        # Find mean score and then get bias of individual items respect to total-mean
        for i, item_column in enumerate(self.aim_transpose):
            sum = item_column.sum()
            num_score = item_column.count_nonzero()
            if num_score > 0:
                item_means[i] = (sum / num_score) - total_mean
            else:
                item_means[i] = 0.0
        return total_mean, user_means, item_means

    def do_mf(self):

        previous_loss = 99999.0
        if self.user_matrix is None:
            self.user_matrix = np.random.rand(self.num_users, self.k)
        if self.item_matrix is None:
            self.item_matrix = np.random.rand(self.k, self.num_items)

        for i in range(self.max_iteration):

            # Refresh user-matrix and then item-matrix
            self.partial_optimize(user_to_item=False)
            self.partial_optimize(user_to_item=True)

            # Stop iteration if loss converges
            new_loss = self.calc_loss()
            loss_update = previous_loss - new_loss

            print("Finished #{} iteration".format(i + 1))
            print("RMSE to training data = {}".format(new_loss))

            if loss_update < self.converge_threshold:
                break
            previous_loss = new_loss

    def partial_optimize(self, user_to_item):
        """
        when A = U * I:
        I = (U^T * U)^-1 * U^T * A

        then A^T = I^T U^T:
        U^T = (I * I^T)^-1 * I * A^T
        
        These equations optimize I or U^T in the context of least-squares
        This function just solves equation above
        """
        if user_to_item:
            base = self.user_matrix
            aim = self.aim_csc
        else:
            base = self.item_matrix.transpose()
            aim = self.aim_transpose.tocsc()

        BtB = np.dot(base.transpose(), base)                # let B=base, calc B^T * B
        BtBinvBt = np.dot(la.inv(BtB), base.transpose())    # calc (B^T * B)^-1 * B^T
        target = sp.csc_matrix(BtBinvBt).dot(aim)           # calc (B^T * B)^-1 * B^T * aim

        if user_to_item:
            self.item_matrix = target.toarray()
        else:
            self.user_matrix = target.transpose().toarray()

    def calc_loss(self):
        # Calculate RMSE of current user_matrix * item_matrix and aim_matrix
        rmse = 0.0
        item_matrix_transpose = self.item_matrix.transpose()
        for (row, col) in zip(*self.aim_csr.nonzero()):
            rmse += (self.aim_csr[row, col] - np.dot(self.user_matrix[row], item_matrix_transpose[col])) ** 2
        rmse /= self.aim_csr.count_nonzero()
        rmse = np.sqrt(rmse)
        return rmse

    def recommend(self, user, num=10, exclude_viewed=True):

        # Make predicted score-array of chosen user
        score_array = np.dot(self.user_matrix[user], self.item_matrix) + self.user_means[user] + self.total_mean
        score_array += self.item_means
        item_id_array = np.arange(self.num_items)

        # Sort predicted score-array so it lines in high-score order
        recommend_matrix = zip(item_id_array, score_array)
        recommend_matrix = sorted(recommend_matrix, key=lambda i: i[1], reverse=True)

        # Get list of items which chosen user already scored
        (hoge, user_rated) = self.aim_csr.getrow(user).nonzero()
        user_rated = np.append(user_rated, 0)

        # Make Top-num Ranking of recommending movies-id
        recommend_top = []
        count = 1
        for index in range(self.num_items):
            id = recommend_matrix[index][0]
            if exclude_viewed and id in user_rated:
                continue

            recommend_top.append(id)
            count += 1
            if count > num:
                break

        return recommend_top

    def calc_precision(self, test_ratings):
        # Calculate RMSE of user_matrix * item_matrix and test data
        rmse = 0.0
        item_matrix_transpose = self.item_matrix.transpose()
        for (user_id, item_id, score) in test_ratings:
            predicted_score = np.dot(self.user_matrix[user_id], item_matrix_transpose[item_id]) + self.total_mean +\
                              self.user_means[user_id] + self.item_means[item_id]
            rmse += (score - predicted_score) ** 2
        rmse /= len(test_ratings)
        rmse = np.sqrt(rmse)
        return rmse