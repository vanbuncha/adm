import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.linear_model import LinearRegression
class recommenderSystem():


    def Naive_1(self, train_df, test_df):
        # Naive Approach
        r_item = train_df.groupby('MovieID')['Rating'].mean().reset_index().rename({'Rating':
                                                                        'R_item'},axis='columns')
        
        r_user = train_df.groupby('UserID')['Rating'].mean().reset_index().rename({'Rating':
                                                                        'R_user'},axis='columns')

        train_df=train_df.merge(r_item, on=['MovieID']).merge(r_user, on=['UserID'])

        #Handle instances were we do not have instances in the training set of movies/users in test set
        test_df=test_df.merge(r_item, on=['MovieID']).merge(r_user, on=['UserID'])
        test_only_users = set(test_df['user_id']) - set(train_df['user_id'])
        test_only_movies = set(test_df['movie_id']) - set(train_df['movie_id'])
        global_average_rating = train_df['Rating'].mean()
        for user in test_only_users:
            test_df.loc[test_df['user_id'] == user, 'R_user'] = global_average_rating

        for movie in test_only_movies:
            test_df.loc[test_df['movie_id'] == movie, 'R_item'] = global_average_rating


        X = train_df[['R_item','R_user']]
        y = train_df['Rating']
        model = LinearRegression().fit(X, y)

        alpha, beta = model.coef_
        gamma = model.intercept_

        X_test = test_df[['R_item','R_user']]
        y_test = test_df["Rating"]
        
        # Predict ratings for the test set
        y_pred = model.predict(X_test)

        # Calculate the root mean squared error (RMSE) for the predictions
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Print the coefficients and RMSE
        print(f"Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")
        print(f"Root Mean Squared Error: {rmse}")
        
        return train_df, test_df

    def matrix_factorization(self, train_df, test_df):
        # The Matrix Factorization

        num_factors=10
        num_iter=75
        reg=0.05
        lr=0.005
        num_users = df_ratings['UserID'].nunique()
        num_movies = df_ratings['MovieID'].nunique()
        
        U = np.random.rand(num_users, num_factors)
        V = np.random.rand(num_movies, num_factors).T
        R = pd.pivot_table(train_df, index='UserID', columns='MovieID', values='Rating').values
        for i in range(num_iter):
            for i in range(num_users):
                for j in range(num_movies):
                    error = R[i][j] –  np.dot(U[i, :], V[:, j])
                    for k in range(num_factors):
                        U[i][k] += lr * (error * V[k][j] - reg*  U[i][k])
                        V[k][j] += lr * (error * U[i][k] - reg*  V[k][j])
            # Compute loss
            total_error = 0
            for i in range(num_users):
                for j in range(num_movies):
                    if R[i][j] > 0:
                        total_error += (R[i][j] - np.dot(U[i, :], V[:, j]))**2

            print(f"Total Training Error epoch {e}: {total_error}")
       # Test the model on the test set and compute the RMSE
            test_matrix = pd.pivot_table(test_df, index='UserID', columns='MovieID', values='Rating').values
            test_error = 0
            count = 0
            
            for i in range(num_users):
                for j in range(num_movies):
                    if not np.isnan(test_matrix[i][j]):
                        pred = np.dot(U[i, :], V[:, j])
                        test_error += (test_matrix[i][j] - pred)**2
                        count += 1
            
            if count > 0:
                test_rmse = sqrt(test_error / count)
                print(f"Iteration {iteration+1}: Test RMSE = {test_rmse}")
        
    def visualisation_1(self):
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        
    def visualisation_2(self):
        # Apply t-SNE
        tsne = TSNE(n_components=2, verbose=1)
        tsne_result = tsne.fit_transform(data)
        
    def visualisation_3(self):
        # Apply UMAP
        umap_model = umap.UMAP(n_components=2)
        umap_result = umap_model.fit_transform(data)

    def cross_validation(self,df_ratings, folds, model):
        # prepare cross validation
        # Shuffle DataFrame
        df_ratings = df_ratings.sample(frac=1).reset_index(drop=True)

        # Split DataFrame into folds
        num_rows = len(df_ratings)
        fold_size = num_rows // folds
        splits = []

        for i in range(folds):
            start_index = i * fold_size
            end_index = (i + 1) * fold_size if i < folds - 1 else num_rows
            test_df = df_ratings.iloc[start_index:end_index]
            train_df = pd.concat([df_ratings.iloc[:start_index], df_ratings.iloc[end_index:]])
            if (model=="Naive"):
                train_df, test_df=self.Naive_1(train_df, test_df)
            elif (model=="Matrix"):
                train_df, test_df = self.matrix_factorization(train_df, test_df)
    
    def perf_measures(y_true,y_pred):
        # Calculate RMSE (Root Mean Squared Error)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print(f'RMSE: {rmse}')

        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)
        print(f'MAE: {mae}')
        
    def main():
        train_list,test_list=self.cross_validartion(5);
        
            
        
if __name__ == '__main__':


            # Specify the file path
        file_path = '/Users/markrademaker/Downloads/ml-1m/ratings.dat'
        df_ratings = pd.read_csv(file_path, sep='::',header=None, engine='python')
        df_ratings = df_ratings.rename({0: 'UserID',
                                        1:'MovieID',
                                        2:'Rating',
                                        3:'Timestamp'},axis='columns')

        print(df_ratings.head())
        # Specify the file path
        file_path = '/Users/markrademaker/Downloads/ml-1m/users.dat'
        df_users = pd.read_csv(file_path, sep='::',header=None, engine='python')
        df_users = df_users.rename({0: 'UserID',
                                        1:'Gender',
                                        2:'Age',
                                        3:'Occupation',
                                        4: 'Zip-code'
                                        },axis='columns')
        print(df_users.head())
        # Specify the file path
        file_path = '/Users/markrademaker/Downloads/ml-1m/movies.dat'
        df_movies = pd.read_csv(file_path, sep='::', header=None, encoding='ISO-8859-1', engine='python')
        df_movies = df_movies.rename({0: 'MovieID',
                                        1:'Title',
                                        2:'Genre'},axis='columns')
        

        
        print(df_movies.head())
        rec= recommenderSystem();
        df_filled=rec.Naive_1(df_ratings);
        
        rec.matrix_factorization(df_filled);
