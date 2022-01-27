"""A module for naive Bayes classifiers"""
import numpy as np


class GaussianNBClassifier:
    def __init__(self, eps=1e-6):
        r"""
    
        """
        self.labels = None
        self.parameters = {
            "li": None,  # shape: (K, M)
            "sigma": None,  # shape: (K, M)
            "prior": None,  # shape: (K,)
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
        H = self.hyperparameters

        self.labels = np.unique(y)

        K = len(self.labels)
        N, M = X.shape

        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))

        for i, c in enumerate(self.labels):
            X_c = X[y == c, :]

            P["mean"][i, :] = np.mean(X_c, axis=0)
            P["sigma"][i, :] = np.var(X_c, axis=0) + H["eps"]
            P["prior"][i] = X_c.shape[0] / N
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
        return self.labels[self._log_posterior(X).argmax(axis=1)]

    def _log_posterior(self, X):
        r"""
        Compute the (unnormalized) log posterior for each class.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        log_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            Unnormalized log posterior probability of each class for each
            example in `X`
        """
        K = len(self.labels)
        log_posterior = np.zeros((X.shape[0], K))
        for i in range(K):
            log_posterior[:, i] = self._log_class_posterior(X, i)
        return log_posterior

    def _log_class_posterior(self, X, class_idx):
        r"""
        Compute the (unnormalized) log posterior for the label at index
        `class_idx` in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`.

        Notes
        -----
        Unnormalized log posterior for example :math:`\mathbf{x}_i` and class
        :math:`c` is::

        .. math::

            \log P(y_i = c \mid \mathbf{x}_i, \theta)
                &\propto \log P(y=c \mid \theta) +
                    \log P(\mathbf{x}_i \mid y_i = c, \theta) \\
                &\propto \log P(y=c \mid \theta)
                    \sum{j=1}^M \log P(x_j \mid y_i = c, \theta)

        In the Gaussian naive Bayes model, the feature likelihood for class
        :math:`c`, :math:`P(\mathbf{x}_i \mid y_i = c, \theta)` is assumed to
        be normally distributed

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`
        class_idx : int
            The index of the current class in :py:attr:`labels`

        Returns
        -------
        log_class_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            Unnormalized log probability of the label at index `class_idx`
            in :py:attr:`labels <numpy_ml.linear_models.GaussianNBClassifier.labels>`
            for each example in `X`
        """  # noqa: E501
        P = self.parameters
        mu = P["mean"][class_idx]
        prior = P["prior"][class_idx]
        sigsq = P["sigma"][class_idx]

        # log likelihood = log X | N(mu, sigsq)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
        log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
        return log_likelihood + np.log(prior)