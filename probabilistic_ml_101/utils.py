import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

def graph_limits(dataset_split):
    """
    Returns the x-axis and y-axis limits for the dataset
    """
    X_train, X_test, y_train, y_test = dataset_split

    # x-axis limits
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
    # y-axis limits (not y the category!)
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5

    return x_min, x_max, y_min, y_max

def scatter_datapoints(ax, dataset_split):
    X_train, X_test, y_train, y_test = dataset_split
    cmap = ListedColormap(['r', 'b'])

    # Plot the training points (as regular points)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, label='train', edgecolors='k')
    # Plot the testing points (as '*')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, s=300, marker='*', label='test', edgecolors='k')

def plot_dataset(dataset_split, fname=''):
    figure, ax = plt.subplots(nrows=1, ncols=1)

    x_min, x_max, y_min, y_max = graph_limits(dataset_split)

    scatter_datapoints(ax, dataset_split)

    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()
    plt.close()

def compare_classifiers_plot(dataset_split,
                             classifiers,
                             fname='',
                             mesh_step=0.02,
                             verbose=False,
                             ):
    """
    Plot decision boundaries for a set of classifiers and compare their respective accuracies
    """
    X_train, X_test, y_train, y_test = dataset_split

    fig = plt.figure(figsize=(13, 6))
#     plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    plt.axis('off')

    nrows = 1
    ncols = len(classifiers)

    clf_names = [clf.name for clf in classifiers]

    for (i, (clf_name, clf)) in enumerate(zip(clf_names, classifiers)):
        info_fit = clf.fit(X_train, y_train)

        if verbose:
            print(info_fit, '\n')
            
        y_pred_in = clf.predict(X_train)
        y_pred = clf.predict(X_test)

        ax = plt.subplot(nrows, ncols, i+1)

        # show datapoints
        scatter_datapoints(ax, dataset_split)

        x_min, x_max, y_min, y_max = graph_limits(dataset_split)

        x_grid = np.arange(x_min, x_max, mesh_step)
        y_grid = np.arange(y_min, y_max, mesh_step)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh based on the magnitude of the decision function.
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.6)

        in_acc = accuracy_score(y_train, y_pred_in)
        out_acc = accuracy_score(y_test, y_pred)

        # print accuracy of each classifier, and in case of a cv random search,
        # the value of the optimal hyperparameters
        xlabel ='{}\nIn: {:.2%}\nOut: {:.2%}'.format(clf_name, in_acc, out_acc)

        if hasattr(clf, "decision_boundary"):
            ax.plot(x_grid, clf.decision_boundary(x_grid), label='boundary', color='r')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.xlabel(xlabel)
        ax.legend(loc='upper right')

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()

def confusion_matrix(y_pred, y_truth):
    """
    Returns number of true/false positives/negatives.
    """
    tp = np.sum(np.multiply(y_pred, y_truth))
    tn = np.sum(np.multiply(1-y_pred, 1-y_truth))
    fp = np.sum(np.multiply(y_pred, 1-y_truth))
    fn = np.sum(np.multiply(1-y_pred, y_truth))

    return tp, tn, fp, fn

def ROC(dataset_split,
        classifiers,
        mesh_number=500,
        threshold_min=-2.0,
        threshold_max=2.0,
        fname='',
        ):
    """
    Compute and plot the ROC curve for classifier clf.
    """
    X_train, X_test, y_train, y_test = dataset_split

    fig = plt.figure(figsize=(13, 6))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    nrows = 1
    ncols = len(classifiers)

    clf_names = [clf.name for clf in classifiers]

    for (i, (clf_name, clf)) in enumerate(zip(clf_names, classifiers)):
        if not hasattr(clf, 'threshold'):
            raise Exception('ROC only available for classifiers with threshold.')

        clf.fit(X_train, y_train)

        thresholds = np.linspace(threshold_min, threshold_max, mesh_number)
        # add extreme threshold to make sure the ROC curve goes through
        # (0 fpr, 1 tpr) and (1 tpr, 0 fpr).
        thresholds[0] = -1000
        thresholds[-1] = 1000

        tpr = []
        fpr = []
        for threshold in thresholds:
            clf.threshold = threshold
            y_pred = clf.predict(X_test)
            tp, tn, fp, fn = confusion_matrix(y_pred, y_test)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))

        auc = np.trapz(tpr[::-1], fpr[::-1])

        ax = plt.subplot(nrows, ncols, i+1)

        ax.plot(fpr, tpr, label='AUC={:.4f}'.format(auc))

        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.title.set_text('ROC for {}'.format(clf.name))
        ax.legend(loc='lower right')

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()

def draw_ellipses(model, ax, colors):
    for k, color in enumerate(colors):
        if model.__class__.__name__ == 'GaussianMixture':
            cov = model.covariances_[k][:2, :2]
            centroids = model.means_[k, :2]
        elif model.__class__.__name__ == 'EMDiagonal':
            cov = model.cov_matrix(k)
            centroids = model.mu[:2, k]
        else:
            raise Exception('{} not an compatible class!'.format(model.__class__.__name__))
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # First axis is given by the first eigenvector
        axis_1 = eigenvectors[0] / np.linalg.norm(eigenvectors[0], ord=2)
        angle = np.arctan2(axis_1[1], axis_1[0]) * 180 / np.pi # in degrees
        
        # we choose to have ellipses correspond to 5 std dev 
        eigenvalues = 5*np.sqrt(eigenvalues)
        
        ell = mpl.patches.Ellipse(centroids, eigenvalues[0], eigenvalues[1],
                                  180 + angle, color=color)
        
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        
def compare_clustering(models, X, y, Ks):
    """
    Input: 
        models: list of sklearn-like clustering classes,
        X: point clouds,
        y: true cluser labels,
        Ks: list of n_clusters to try for each clustering algo.
    """
    cmap = plt.get_cmap('gist_rainbow')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=np.max(Ks))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = scalarMap.to_rgba(range(np.max(Ks)))
                  
    # Ground truth
    fig, ax = plt.subplots(figsize=(4,5), nrows=1, ncols=1)
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=mpl.colors.ListedColormap(colors[:3]))
    ax.set_title('Ground Truth')
    ax.axis('off')
    ax.set_aspect('equal', 'datalim')
    
    nrows = len(Ks)
    ncols = len(models)
    # plot the clustering with differents numbers of clusters
    fig, axess = plt.subplots(figsize=(5*ncols, 4*nrows), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    for i, K in enumerate(Ks):
        # handles the case where we test only a single number of clusters 
        if len(Ks) == 1:
            axes = axess
        else:
            axes = axess[i]
        for j, (name, model) in enumerate(models.items()):
            ax = axes[j]

            if model.__class__.__name__ == 'GaussianMixture':
                model.n_components = K
            elif model.__class__.__name__ in ['EMDiagonal', 'KMeans']:
                model.n_clusters = K
                
            labels = model.fit_predict(X)
            
            colors_K = colors[:K]
            for k, color in enumerate(colors_K):
                ax.scatter(X[:, 0], X[:, 1], c=labels,
                           cmap=mpl.colors.ListedColormap(colors_K))
                if model.__class__.__name__ != 'KMeans':
                    draw_ellipses(model, ax, colors_K)
                ax.set_title(name)
            ax.axis('off')
            ax.set_aspect('equal', 'datalim')

    plt.tight_layout()
    plt.show()

def truncnorm_plus(m):
    """Sample from truncated Gaussian TN(m, 1, [0, infty]) by rejection.
    """
    z = -1
    while z<=0:
        z = np.random.randn()+m
    return z

def truncnorm_minus(m):
    """Sample from truncated Gaussian TN(m, 1, [-infty, 0]) by rejection.
    """
    z = 1
    while z>=0:
        z = np.random.randn()+m
    return z
