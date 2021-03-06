import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Plot 50 random seeds which is scattered around
# a line with slope 2 and interception at -5.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
# plt.title("Random seed generated data")
# plt.scatter(x, y)
# plt.show()

# Create a best fitting line using Scikit-Learns
# linear regression estimator.
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.title("Linear Regression")
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)

print(model.intercept_)
print(model.coef_)

# ---------------- Polynomial Basis Function  ---------------------#

from sklearn.preprocessing import PolynomialFeatures

x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
xfit = np.linspace(0, 10, 1000)
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.title("7th order Polynomial Regression")
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

# ---------------- Guassian Basis Function ---------------------#

from sklearn.base import BaseEstimator, TransformerMixin


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y = None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.title("Gussian Basis Function using 20-dimensions")
plt.plot(xfit, yfit)
plt.xlim(0, 10)
plt.show()

gussian_model_of = make_pipeline(GaussianFeatures(30), LinearRegression())
gussian_model_of.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.title("Gussian Basis Function using 30-dimensions (Overfitted)")
plt.plot(xfit, gussian_model_of.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.show()