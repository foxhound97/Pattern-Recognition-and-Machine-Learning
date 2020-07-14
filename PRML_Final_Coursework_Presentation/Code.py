import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import scale

class Sampler(object):
	def __init__(self, n, p = None, d = None):
		"""
		@param n: number of samples
		@param p: probability of assigning Y=1, p > 1/2
		@param d: data dimension d >= 6n
		"""
		assert p is None or p > 0.5
		assert d is None or d > 6*n
		self.n = n
		if p is None:
			p = 0.0
			while p <= 0.5:
				p = np.random.rand()
		self.p = p
		self.d = 6*n+1 if d is None else d

	def sample(self):
		probs = np.random.rand(self.n)
		Y = np.ones(self.n).astype(np.int)
		neg_idx = np.where(probs > self.p)
		Y[neg_idx] = -1
		X = np.zeros((self.n, self.d), np.float)
		X[:,0] = Y
		X[:,[1,2]] = 1
		for i in range(self.n):
			start_idx = 3+5*i
			end_idx = start_idx + 2*(1-Y[i]) + 1
			X[i,start_idx:end_idx] = 1
		return X, Y

class LinearRegression(object):
	def __init__(self, batch_size, nepochs, lr, lr_decay=1.0, hb=0.0, nesterov=0.0, method="SGDOptimizer", eps=1e-6, decay=0.0):
		assert method in ["SGDOptimizer", "AdagradOptimizer"]
		self.batch_size = batch_size
		self.nepochs = nepochs
		self.lr = lr
		self.lr_decay = lr_decay
		self.hb = hb
		self.nesterov = nesterov
		self.method = method
		self.eps = eps
		self.decay = decay

	def train(self, X, y, X_test=None, y_test=None):
		n, p = X.shape
		if self.method == "SGDOptimizer":
			self.w = np.zeros(p) # initialize with all 0's with SGD
		else:
			ub = np.sqrt(12/p)
			self.w = np.random.uniform(-ub, ub, p) # Xavier initialization with Adagrad
			#self.w = np.random.normal(0, ub, p)
			#self.w = np.zeros(p)
		self.loss_history = []
		self.test_loss_history = []
		loss = self.evaluate(X, y)
		print("Begin training. Initial loss is {:.3f}".format(loss))
		self.loss_history.append(loss)
		for e in range(self.nepochs):
			if self.method == "SGDOptimizer":
				loss = self.train_SGD_epoch(e, X, y)
			else:
				loss = self.train_Adagrad_epoch(e, X, y)
			loss = self.evaluate(X, y)
			print("Epoch {} completed! Current loss is {:.3f}".format(e+1, loss))
			if X_test is not None:
				test_loss = self.evaluate(X_test, y_test)
				self.test_loss_history.append(test_loss)
			self.loss_history.append(loss)
			self.lr *= self.lr_decay

	def train_Adagrad_epoch(self, e, X, y):
		n, p = X.shape
		iter_indices = np.random.permutation(n)
		for i in range(0, n, self.batch_size):
			batch_indices = iter_indices[i:i+self.batch_size]
			X_batch, y_batch = X[batch_indices], y[batch_indices]
			if e == 0 and i == 0:
				grad = self.calculate_gradient(X_batch, y_batch, self.w)
				self.G = grad * grad + self.eps
				self.H = np.sqrt(self.G)
				self.past_w = copy.deepcopy(self.w)
				self.w = self.w - self.lr * (1 / self.H) * grad + self.decay * self.w
			else:
				grad = self.calculate_gradient(X_batch, y_batch, self.w + self.nesterov * (self.w - self.past_w))
				self.G += grad * grad
				self.past_H = copy.deepcopy(self.H)
				self.H = np.sqrt(self.G)
				temp_w = copy.deepcopy(self.w)
				self.w = self.w - self.lr * (1 / self.H) * grad + self.hb * (1 / self.H) * self.past_H * (self.w - self.past_w) + self.decay * self.w
				self.past_w = temp_w

	def train_SGD_epoch(self, e, X, y):
		n, p = X.shape
		iter_indices = np.random.permutation(n)
		for i in range(0, n, self.batch_size):
			batch_indices = iter_indices[i:i+self.batch_size]
			X_batch, y_batch = X[batch_indices], y[batch_indices]
			if e == 0 and i == 0:	
				grad = self.calculate_gradient(X_batch, y_batch, self.w)
				self.past_w = copy.deepcopy(self.w)
				self.w = self.w - self.lr * grad + self.decay * self.w
			else:
				grad = self.calculate_gradient(X_batch, y_batch, self.w + self.nesterov * (self.w - self.past_w))
				temp_w = copy.deepcopy(self.w)
				self.w = self.w - self.lr * grad + self.hb * (self.w - self.past_w) + self.decay * self.w
				self.past_w = temp_w

	def calculate_gradient(self, X, y, w):
		grad = 2 * np.matmul(X.T, (np.matmul(X, self.w)) - y) / self.batch_size
		return grad

	def evaluate(self, X, y):
		return np.sum((y - np.matmul(X, self.w)) ** 2) / X.shape[0]

	def yhat(self, X):
		return np.matmul(X,self.w)

if __name__ == "__main__":

	num_samples = 2000
	s = Sampler(num_samples, 0.8)
	X, y = s.sample()
	XTrain, yTrain = X[:int(num_samples*0.7)], y[:int(num_samples*0.7)]
	XTest, yTest = X[int(num_samples*0.7):], y[int(num_samples*0.7):]
	batch_sizes = [1, 16, 64]
	lrs = [1e-3]
	methods = ["SGDOptimizer","AdagradOptimizer"]
	for b, l, m in product(batch_sizes, lrs, methods):
		lr = LinearRegression(b, 200, l, method=m)
		lr.train(XTrain, yTrain, XTest, yTest)
		plt.plot(lr.test_loss_history, label="-".join([m[:-9], str(b)]))
	plt.xlabel("Epoch")
	plt.ylabel("Test Loss")
	plt.title("Results")
	plt.legend()
	plt.show()
	


