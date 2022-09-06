import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, cluster, neighbors
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import OneHotEncoder


class Compressed_kNN:


    def __init__(self, n_cols=100, n_rows=500, n_neighbors=5, whiten=False,
                 minibatch_size_pca=10000, minibatch_size_kmeans=5000, show_plots=False, verbose=0, random_state=1):
        """TextureModel = TextureModel_PCA(n_components, minibatch_size, random_state)"""
        self.n_cols      = n_cols
        self.n_rows      = n_rows
        self.n_neighbors = n_neighbors
        self.whiten = whiten
        self.minibatch_size_pca = minibatch_size_pca
        self.minibatch_size_kmeans = minibatch_size_kmeans
        self.random_state = random_state
        self.show_plots   = show_plots
        self.verbose      = verbose


    def fit(self, X, y):
        """Compressed_kNN.fit(X, y)"""
        if self.verbose > 0:
            print('---------------------------------------------------------------------')
        (num_samples, num_features_x) = X.shape

        if len(y.shape) < 2:
            y = y[:,np.newaxis]

        # PCA to compress columns
        n_components = min(self.n_cols, int(0.9 * num_features_x), int(0.9 * num_samples))

        if num_samples < 200000:
            PCA_cols_compressor = decomposition.PCA(n_components=n_components, whiten=self.whiten)
        else:
            minibatch_size_pca = min(self.minibatch_size_pca, num_samples)
            PCA_cols_compressor = decomposition.IncrementalPCA(n_components=n_components, batch_size=minibatch_size_pca, whiten=self.whiten)
        PCA_cols_compressor.fit(X)

        explained_variance_ratio_pca = PCA_cols_compressor.explained_variance_ratio_.sum()
        if self.verbose > 0:
            print('explained var by PCA    model with %4d components is %.1f%s' %(n_components, 100 * explained_variance_ratio_pca,'%'))

        if self.show_plots:
            plt.figure(); plt.plot(1 + np.arange(PCA_cols_compressor.n_components), 100 * np.cumsum(PCA_cols_compressor.explained_variance_ratio_))
            plt.xlabel('num components'); plt.ylabel('explained %s' %('%')); plt.title('cumulative explained percent')

        self.PCA_cols_compressor = PCA_cols_compressor


        # k-means to compress rows
        X_reduced_cols = PCA_cols_compressor.transform(X)

        n_clusters = min(self.n_rows, int(0.8 * self.minibatch_size_kmeans), int(0.8 * num_samples))

        if num_samples < 20000:
            KMeans_rows_compressor = cluster.KMeans(n_clusters=n_clusters)
        else:
            minibatch_size_kmeans = int(min(self.minibatch_size_kmeans, num_samples))
            KMeans_rows_compressor = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=minibatch_size_kmeans)
        KMeans_rows_compressor.fit(X_reduced_cols)

        explained_variance_ratio_kmeans = 1 - (KMeans_rows_compressor.inertia_ / ((X_reduced_cols - X_reduced_cols.mean(axis=0)) ** 2).sum())
        if self.verbose > 0:
            print('explained var by KMeans model with %4d templates  is %.1f%s' %(n_clusters, 100 * explained_variance_ratio_kmeans, '%'))

        self.KMeans_rows_compressor = KMeans_rows_compressor

        # k-nearest neighbors
        template_inds = KMeans_rows_compressor.predict(X_reduced_cols)

        X_reduced_rows_cols = np.zeros((self.KMeans_rows_compressor.n_clusters, self.PCA_cols_compressor.n_components))
        y_reduced_rows      = np.zeros((self.KMeans_rows_compressor.n_clusters, y.shape[1]))

        for template_ind in template_inds:
            X_reduced_rows_cols[template_ind] = X_reduced_cols[template_inds == template_ind].mean(axis=0)
            y_reduced_rows[template_ind]      = y[template_inds == template_ind].mean(axis=0)

        n_neighbors = min(self.n_neighbors, self.n_rows, num_samples)
        kNN_mapper = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
        kNN_mapper.fit(X_reduced_rows_cols, y_reduced_rows)

        self.kNN_mapper = kNN_mapper

        var_retained = 100 * explained_variance_ratio_pca * explained_variance_ratio_kmeans
        numbers_reduced = 100 * (X_reduced_rows_cols.shape[0] * X_reduced_rows_cols.shape[1]) / (X.shape[0] * X.shape[1])
        if self.verbose > 0:
            print('finished training')
            print('-------------------')
            print('compressed from X.shape = %s to X_reduced.shape = %s' %(str(X.shape), str(X_reduced_rows_cols.shape)))
            print('in total, retained %.2f%s of the variance with %.2f%s of the numbers' %(var_retained, '%', numbers_reduced, '%'))
            print('---------------------------------------------------------------------')


    def predict(self, X):
        '''y_hat = Compressed_kNN.predict(X)'''

        X_reduced_cols = self.PCA_cols_compressor.transform(X)
        y_hat = self.kNN_mapper.predict(X_reduced_cols)

        return y_hat


if __name__ == "__main__":
    # load sample dataset
    olivetti_faces_dict = fetch_olivetti_faces(shuffle=True, random_state=42)

    faces = olivetti_faces_dict['images']
    class_labels = olivetti_faces_dict['target'].reshape(-1,1)

    # translate outputs to proper format
    class_ecnoder = OneHotEncoder(handle_unknown='ignore')
    class_ecnoder.fit(class_labels)
    y = class_ecnoder.transform(class_labels).toarray()

    # translate inputs to proper format
    X = np.reshape(faces, [faces.shape[0], -1])

    # define and train model
    train_inds = np.arange(300)
    valid_inds = np.arange(300,400)

    kNN_classfier = Compressed_kNN(n_cols=60, n_rows=120, n_neighbors=1, whiten=False,
                                   minibatch_size_pca=10000, minibatch_size_kmeans=1000, show_plots=False)
    kNN_classfier.fit(X[train_inds],y[train_inds])

    # evaluate performace
    y_train_hat = kNN_classfier.predict(X[train_inds])
    y_valid_hat = kNN_classfier.predict(X[valid_inds])

    train_Acc = (np.argmax(y_train_hat, axis=1) == np.argmax(y[train_inds], axis=1)).mean()
    valid_Acc = (np.argmax(y_valid_hat, axis=1) == np.argmax(y[valid_inds], axis=1)).mean()

    train_RMSE = np.sqrt(((y_train_hat - y[train_inds]) ** 2).mean())
    valid_RMSE = np.sqrt(((y_valid_hat - y[valid_inds]) ** 2).mean())

    print('Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc, valid_Acc))
    print('RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE, valid_RMSE))
