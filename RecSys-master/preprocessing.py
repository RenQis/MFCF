import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def read_data():
    ratings_headers = ['user_id', 'movie_id', 'rating']
    df = pd.read_csv('data/dataset.csv', header=None, names=ratings_headers)
    ratings = df.copy()
    ratings_org = df.copy()
    return df, ratings, ratings_org

def extract(ratings):
    sub_ratings = ratings.loc[random.sample(list(ratings.index), 40)]
    ratings = ratings.drop(sub_ratings.index)
    return ratings, sub_ratings

def create_matrix(df, ratings):
    for i in ratings.index:
        df.ix[i]['rating'] = ratings.ix[i]['rating']
    ratings = df.copy()
    df['rating'] = 0
    return df, ratings

def pivot(ratings):
    ratings = pd.pivot_table(ratings, values='rating', index=['user_id'], columns=['movie_id'])
    ratings = ratings.fillna(0)
    ratings = ratings.as_matrix()
    return ratings

def split(ratings1, ratings2, ratings3, ratings4, ratings5):
    dataset1 = (ratings1 + ratings2 + ratings3 + ratings4, ratings5)
    dataset2 = (ratings1 + ratings2 + ratings3 + ratings5, ratings4)
    dataset3 = (ratings1 + ratings2 + ratings4 + ratings5, ratings3)
    dataset4 = (ratings1 + ratings3 + ratings4 + ratings5, ratings2)
    dataset5 = (ratings2 + ratings3 + ratings4 + ratings5, ratings1)
    return dataset1, dataset2, dataset3, dataset4, dataset5

def create_datasets():
    df, ratings, ratings_org = read_data()
    df['rating'] = 0
    print(type(ratings))
    ratings, ratings1 = extract(ratings)
    ratings, ratings2 = extract(ratings)
    ratings, ratings3 = extract(ratings)
    ratings, ratings4 = extract(ratings)
    ratings, ratings5 = extract(ratings)
    #print(ratings, ratings1)

    df, ratings1 = create_matrix(df, ratings1)
    df, ratings2 = create_matrix(df, ratings2)
    df, ratings3 = create_matrix(df, ratings3)
    df, ratings4 = create_matrix(df, ratings4)
    df, ratings5 = create_matrix(df, ratings5)

    ratings_org = pivot(ratings_org)
    ratings1 = pivot(ratings1)
    ratings2 = pivot(ratings2)
    ratings3 = pivot(ratings3)
    ratings4 = pivot(ratings4)
    ratings5 = pivot(ratings5)
    #print(ratings1)
    return split(ratings1, ratings2, ratings3, ratings4, ratings5)
