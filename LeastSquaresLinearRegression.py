'''
Test Case
---------
>>> prng = np.random.RandomState(0)
>>> N = 100

>>> true_w_F = np.asarray([1.1, -2.2, 3.3])
>>> true_b = 0.0
>>> x_NF = prng.randn(N, 3)
>>> y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

>>> linear_regr = LeastSquaresLinearRegressor()
>>> linear_regr.fit(x_NF, y_N)

>>> yhat_N = linear_regr.predict(x_NF)
>>> np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})
>>> print(linear_regr.w_F)
[ 1.099 -2.202  3.301]
>>> print(np.asarray([linear_regr.b]))
[-0.005]
'''

import numpy as np
# No other imports allowed!

class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
        
        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        '''      
        N, F = x_NF.shape

        ones_column = np.ones((N, 1))

        # Note: G = F + 1
        x_NG = np.hstack((x_NF, ones_column))

        xTx_GG = np.dot(x_NG.T, x_NG)

        xTy_GN = np.dot(x_NG.T, y_N)

        theta = np.linalg.solve(xTx_GG, xTy_GN)

        self.b = theta[F]
        self.w_F = theta[:-1]


    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_MF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        '''

        yhat_M = self.b + np.dot(x_MF, self.w_F.T)

        return np.asarray(yhat_M)




def test_on_toy_data(N=100):
    '''
    Simple example use case
    With toy dataset with N=100 examples
    created via a known linear regression model plus small noise
    '''
    prng = np.random.RandomState(0)

    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)

    yhat_N = linear_regr.predict(x_NF)

    np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})

    print("True weights")
    print(true_w_F)
    print("Estimated weights")
    print(linear_regr.w_F)

    print("True intercept")
    print(np.asarray([true_b]))
    print("Estimated intercept")
    print(np.asarray([linear_regr.b]))

if __name__ == '__main__':
    test_on_toy_data()
