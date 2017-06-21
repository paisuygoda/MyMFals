# coding:utf-8
import scipy.sparse as sp
from MyMFals import MyMFals
"""
データファイルの読み込みや結果の表示など
直接Matrix Factorizationと関係ない処理はこちらで請け負う
"""
data_path = "ml-100k/u.data"
item_path = "ml-100k/u.item"
test_train_ratio = 0.1
recommending_user_id = 100

# Get list of user-item-score from file
ratings = []
num_users = -1
num_items = -1
for line in open(data_path, "r"):
    a = line.strip().split()
    user_id = int(a[0])
    item_id = int(a[1])
    score = float(a[2])
    ratings.append((user_id, item_id, score))
    num_users = max(num_users, user_id)
    num_items = max(num_items, item_id)
num_users += 1
num_items += 1

# Create user-score matrix
aim_matrix = sp.lil_matrix((num_users, num_items))
num_test = int(len(ratings) * test_train_ratio)
test_ratings = []
count = 0
for rating in ratings:
    if count < num_test:
        test_ratings.append(rating)
    else:
        aim_matrix[rating[0], rating[1]] = rating[2]
    count += 1

# Do matrix factorization
mf = MyMFals(aim_matrix, num_users, num_items)
mf.do_mf()

# Show precision
print("-------------------------")
print("Finished MF...RMSE to test data = {}".format(mf.calc_precision(test_ratings)))
print("-------------------------")

# Get Top-10 Ranking of recommending movie-id
recommend_top = mf.recommend(recommending_user_id)

# Create list of movie titles
item_title_list = ["ERROR"]
for line in open(item_path, "r"):
    a = line.strip().split("|")
    item_title_list.append(a[1])

# Show recommendation ranking
print("--Movies You Would Like--")
for rank, id in enumerate(recommend_top):
    print("{0}|{1}".format(rank+1, item_title_list[id]))
print("-------------------------")