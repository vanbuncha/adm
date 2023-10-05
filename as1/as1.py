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


    def Naive_1(self, df_ratings):
        # Naive Approach
        r_global = df_ratings['Rating'].mean()
        r_item = df_ratings.groupby('MovieID')['Rating'].mean().reset_index().rename({'Rating':
                                                                                      'R_item'},axis='columns')
        
        r_user = df_ratings.groupby('UserID')['Rating'].mean().reset_index().rename({'Rating': 'R_user'},axis='columns')
        df_ratings=df_ratings.merge(r_item, on=['MovieID']).merge(r_user, on=['UserID'])
        print(r_global)
        print(r_item.head())
        print(r_user.head())
        print(df_ratings.head())

        X = df_ratings[['R_item','R_user']]
        y = df_ratings['Rating']
        model = LinearRegression().fit(X, y)

        alpha, beta = model.coef_
        gamma = model.intercept_

        print(f'alpha: {alpha}, beta: {beta}, gamma: {gamma}')
        
    def function_3(self, train):
        print("hi")
        # The UV matrix decomposition

    def function_4(self, train):
        # The Matrix Factorization
        print("yo")
        
    def function_5(self, train):
        print("world")
        
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

    def cross_validation(self,folds):
        # prepare cross validation
        kfold = KFold(folds, True, 1)
        train_list=[]
        test_list=[]
        # enumerate splits
        for train, test in kfold.split(self.data):
         train_list.append(data[train])
         test_list.append(data[test])
        return train_list, test_list
    
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
        file_path = 'ml-1m/ratings.dat'
        df_ratings = pd.read_csv(file_path, sep='::',header=None, engine='python')
        df_ratings = df_ratings.rename({0: 'UserID',
                                        1:'MovieID',
                                        2:'Rating',
                                        3:'Timestamp'},axis='columns')

        print(df_ratings.head())
        # Specify the file path
        file_path = 'ml-1m/users.dat'
        df_users = pd.read_csv(file_path, sep='::',header=None, engine='python')
        df_users = df_users.rename({0: 'UserID',
                                        1:'Gender',
                                        2:'Age',
                                        3:'Occupation',
                                        4: 'Zip-code'
                                        },axis='columns')
        print(df_users.head())
        # Specify the file path
        file_path = 'ml-1m/movies.dat'
        df_movies = pd.read_csv(file_path, sep='::', header=None, encoding='ISO-8859-1', engine='python')
        df_movies = df_movies.rename({0: 'MovieID',
                                        1:'Title',
                                        2:'Genre'},axis='columns')


        print(df_movies.head())

        rec= recommenderSystem()
        rec.Naive_1(df_ratings)
