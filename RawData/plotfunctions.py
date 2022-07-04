# auxiliary functions for PCA with spectral data
import matplotlib.pyplot as plt
import numpy as np

def scree_plot(PCA, ax=plt):
    """
    In:
        PCA : sklearn.decomposition.PCA
            Fitted PCA object
    Out:
        fig : matplotlib.pyplot.figure object
            the scree plot object
    """

    expl_var_1 = PCA.explained_variance_ratio_

    with plt.style.context(('ggplot')):
        
        
    
        ax.plot(expl_var_1,'-o', label="Explained Variance %")
        ax.plot(np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %')
        ax.set_xlabel("PC number")
        ax.set_ylabel('Variance %')
        ax.set_title('Scree plot')
        
        ax.legend()
        #plt.show()

    


def scores_plot(scores, y, PCs=(1,2), Title='Scores Plot'):
    """
    In:
        scores : np.array of shape (Nrows, Ncomp)
            scores matrix
        y : np.array of shape (Nrows,)
            vector of class labels; 
        PCs [optional] :  2-tuple of integers
            Principle Components to be plotted
    Out:
        fig : matplotlib.pyplot.figure object
            scores plot object
    """
    
    unique = np.unique(y) # list of unique labels
    colors = [plt.cm.jet(float(i)/len(unique)) for i in range(len(unique))]
    
    PCx = PCs[0]
    PCy = PCs[1]
    
    fig = plt.figure()

    with plt.style.context(('ggplot')):
        for i, u in enumerate(unique):
            col = np.expand_dims(np.array(colors[i]), axis=0)
            xi = [scores[j,PCx-1] for j in range(len(scores[:,PCx-1])) if y[j] == u]
            yi = [scores[j,PCy-1] for j in range(len(scores[:,PCy-1])) if y[j] == u]
            plt.scatter(xi, yi, c=col, s=60, edgecolors='k',label=str(u))
    
        plt.xlabel('PC'+str(PCx))
        plt.ylabel('PC'+str(PCy))
        plt.legend(unique,loc='lower right')
        plt.title(Title)
        #plt.show()

    return fig


def loading_plot(loadings, dim, PCs=[1], xlabel='wave length [nm]', ax=plt):
    """
    In:
        loadings : np.array of shape (Nrows, Ncomp)
            loadings matrix
        dim : numpy.ndarray of shape (n_features,)
            Wavelength, -number etc.
        PCs [optional] :  list of integers
            Principle Components to be plotted
    Out:
        fig : matplotlib.pyplot.figure object
            loadings plot object
    """
    if ax==plt:
        fig = plt.figure()
    
        
        with plt.style.context(('ggplot')):
            for PC in PCs:
                ax.plot(dim, loadings[:,PC-1], label='PC'+str(PC))

        ax.xlabel(xlabel)
        ax.ylabel('Loading [a.u.]')
        ax.legend()

        return fig
    else:
        with plt.style.context(('ggplot')):
            for PC in PCs:
                ax.plot(dim, loadings[:,PC-1], label='PC'+str(PC))

        ax.xlabel(xlabel)
        ax.ylabel('Loading [a.u.]')
        ax.legend()

def Tsq_Q_plot(X, scores, loadings, conf=0.95, ax=plt):
    """
    T^2-Q-Plot of PCA results
    adapted from https://nirpyresearch.com/outliers-detection-pls-regression-nir-spectroscopy-python/ 
    
    In:
    X : np.array of shape (Nrows, Ncomp)
        data matrix
    scores : np.array of shape (Nrows, Ncomp)
        scores matrix
    loadings : np.array of shape (Nrows, Ncomp)
        loadings matrix
    conf [optional]: float
        confidence level

    Out:
        fig : matplotlib.pyplot.figure object
            T^2-Q plot object
    """

    ncomp = scores.shape[1]

    # residuals ("errors")
    Err = X - np.dot(scores,loadings.T)

    # Calculate Q-residuals (sum over the rows of the error array)
    Q = np.sum(Err**2, axis=1)

    # Calculate Hotelling's T-squared (note that data are normalised by default)
    Tsq = np.sum((scores/np.std(scores, axis=0))**2, axis=1)
    
    from scipy.stats import f
    # Calculate confidence level for T-squared from the ppf of the F distribution
    Tsq_conf =  f.ppf(q=conf, dfn=ncomp, \
                dfd=X.shape[0])*ncomp*(X.shape[0]-1)/(X.shape[0]-ncomp)

    # Estimate the confidence level for the Q-residuals
    i = np.max(Q)+1
    while 1-np.sum(Q>i)/np.sum(Q>0) > conf:
        i -= 1
    Q_conf = i

   

    with plt.style.context(('ggplot')):
        ax.plot(Tsq, Q, 'o')
    
        ax.plot([Tsq_conf,Tsq_conf],[plt.axis()[2],plt.axis()[3]],  '--')
        ax.plot([plt.axis()[0],plt.axis()[1]],[Q_conf,Q_conf],  '--')
        ax.set_xlabel("Hotelling's T-squared")
        ax.set_ylabel('Q residuals')
        ax.set_title('TSQ Plot')
    
    #plt.show()

    return Q, Tsq

def plotspec(X, dim, title = None, ax=plt):
        """Kreiert Plot von Spektren mit definierbarem Titel
         Args:
            X(int): Spektrum daten
            dim: X-Achse
            title(str, optional)
        Returns:
            fig: matplotlib.pyplot.figure object"""
        
        for i in np.arange(len(X)):
                ax.plot(dim, X[i])

        ax.set_title(title)

        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Absorption [a.u.]')
        

def show_values(axs, orient="v", space=.01, hspace=0.3):
    """displays the values on a seaborn barplot
    Args:
        axs=plot witch should be used
        orient(optional)= orientation of barplot, if not specified vertical oreintation is used
        space(optional, default 0.01): space between bar and number
        hspace(optional, default 0.3): horzontal positioning of number
        """
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*hspace)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)    
