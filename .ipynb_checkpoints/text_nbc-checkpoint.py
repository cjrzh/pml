"""A module for naive Bayes classifiers for Text classification"""
"""朴素贝叶斯二元文本分类器 """
import numpy as np


class TextNBClassifier:
    def __init__(self, eps=1e-6):
        r"""
    
        """
        self.labels = [0,1]  #仅支持0-1两类，仅为学习流程
        self.parameters = {
            "likelihod": None,  # shape: (2, M) 2：2个分类，M: Vacb size 词典大小
            "prior": None,  # shape: (1)
        }

    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.

        Notes
        -----
        The model parameters are stored in the :py:attr:`parameters
        <numpy_ml.linear_models.GaussianNBClassifier.parameters>` attribute.
        The following keys are present:

            "mean": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature means for each of the `K` label classes
            "sigma": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature variances for each of the `K` label classes
            "prior": :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
                Prior probability of each of the `K` label classes, estimated
                empirically from the training data

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`

        Returns
        -------
        self : :class:`GaussianNBClassifier <numpy_ml.linear_models.GaussianNBClassifier>` instance
        """  # noqa: E501
        P = self.parameters
        #H = self.hyperparameters

        #self.labels = np.unique(y)

        #K = len(self.labels) #K=2
        X= np.minimum(X, 1)
        N, M = X.shape

        P["likelihod"] = np.zeros((2, M))
        #训练集非常两类，y=0和y=1 单独计算参数
        X_0 = X[y == 0, :] 
        X_1 = X[y == 1, :]
        P["prior"] = X_1.shape[0] / N #prior 即为y=1的概率

        #y=0,参数为P(x=1 given y=0) ,laplace平滑
        P["likelihod"][0,:]=(np.sum(X_0,axis=0)+1)/(X_0.shape[0]+2)
        P["likelihod"][1,:]=(np.sum(X_1,axis=0)+1)/(X_1.shape[0]+2)
        
        return self

    def predict(self, X):
        """
        Use the trained classifier to predict the class label for each example
        in **X**.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        labels : :py:class:`ndarray <numpy.ndarray>` of shape `(N)`
            The predicted class labels for each example in `X`
        """
        X= np.minimum(X, 1)
        # 计算log P(y=1 give x ) 和log P(y=0 give x) 比较大小
        P = self.parameters
        prior = P["prior"]
        # y=0        
        tmp=(1-X)*(1-P["likelihod"][0,:])+X*P["likelihod"][0,:]  
        p_0=np.sum(np.log(tmp),axis=1)+np.log(prior) #shape(N*1)
        
        #y=1
        tmp=(1-X)*(1-P["likelihod"][1,:])+X*P["likelihod"][1,:]  
        p_1=np.sum(np.log(tmp),axis=1)+np.log(prior) #shape(N*1)

        return np.array([0 if x else 1 for x in p_0>p_1])
