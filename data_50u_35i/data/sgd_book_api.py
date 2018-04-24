from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import os
import io
from sklearn.externals import joblib

app = Flask(__name__)


@app.route("/predict_item", methods=['POST'])
def predict_item():
    if request.method == 'POST':
        try:
            data = request.get_json()
            book_mat = int(data["ItemMat"])

            # sgd_model_item = joblib.load("./python_best_sgd_model.pkl")

            # def cosine_similarity(model):
            #     sim = model.item_vecs.dot(model.item_vecs.T)
            #     norms = np.array([np.sqrt(np.diagonal(sim))])
            #     return sim / norms / norms.T
            #
            # sgd_sim = cosine_similarity(sgd_model_item)

            idx_to_book = {}

            with io.open('reviews_50u_35i_info_items.csv', mode='r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    info = line.split(',')
                    if len(info[1]) <= 10:
                        for asin_char in range(len(info[1]), 11):
                            info[1] = '0' + info[1]

                    idx_to_book[int(info[0]) - 1] = '{"mat_id":' + info[0] + ',"item_id":"' + info[1] + '"}'

            # book_indices = np.argsort(sgd_model_item.sgd_sim[book_mat, :])[::-1]
            book_indices = np.argsort(sgd_model_item.sgd_sim[book_mat, :])[::-1]
            book_idx_list = []
            k_ctr = 0
            # Start i at 1 to not grab the input book
            i = 1
            while k_ctr < 10:
                book = idx_to_book[book_indices[i]]
                book_idx_list.append(book)
                k_ctr += 1
                i += 1

            string_list = str(book_idx_list)
            string_list = string_list.replace("'", "").replace("\\n", "")

        except ValueError:
            return jsonify("Error Predict Item.")

        return jsonify(string_list)


@app.route("/predict_user", methods=['POST'])
def predict_user():
    if request.method == 'POST':
        try:
            data = request.get_json()
            sgd_model_user = joblib.load("./python_best_sgd_model_user.pkl")

            my_ratings = np.zeros((sgd_model_user.item_vecs.shape[0], 1))
            for i in data:
                my_ratings[i['ItemMat']] = i['ItemRating']

            new_user_rec = my_ratings.T.dot(sgd_model_user.item_vecs) \
                .dot(sgd_model_user.item_vecs.T)

            idx_to_book = {}

            with io.open('reviews_50u_35i_info_items.csv', mode='r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    info = line.split(',')
                    if len(info[1]) <= 10:
                        for asin_char in range(len(info[1]), 11):
                            info[1] = '0' + info[1]

                    idx_to_book[int(info[0]) - 1] = '{"mat_id":' + info[0] + ',"item_id":"' + info[1] + '"}'

            book_indices = np.argsort(new_user_rec[0, :], axis=0)[::-1]
            book_idx_list = []
            k_ctr = 0
            i = 0
            while k_ctr < 10:
                book = idx_to_book[book_indices[i]]
                book_idx_list.append(book)
                k_ctr += 1
                i += 1

            string_list = str(book_idx_list)
            string_list = string_list.replace("'", "").replace("\\n", "")

        except ValueError as e:
            return jsonify("Error Predict User")

        return jsonify(string_list)


# @app.route("/retrain", methods=['POST'])
# def retrain():
#     if request.method == 'POST':
#         try:
#             data = request.get_json()


if __name__ == '__main__':
    sgd_model_item = joblib.load("./python_best_sgd_model_item.pkl")
    sgd_model_user = joblib.load("./python_best_sgd_model_user.pkl")

    app.run(host='0.0.0.0')
