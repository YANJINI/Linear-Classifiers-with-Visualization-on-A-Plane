import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
plt.style.use('ggplot')

class twoD_logisticR:
    def __init__(self, learning_rate = 0.01):
        self.count_enter = 0
        self.which_label = 1
        self.lr = learning_rate

        self._background_figure()
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        self.labeled_coordinates = {}
        self.labeled_coordinates[-1] = []
        self.labeled_coordinates[1] = []
        plt.show(block=True)

    def give_labeled_coordinates(self):
        return self.labeled_coordinates

    def _on_click(self, event):
        if self.which_label == 1:
            self._draw_click(event)
            self.labeled_coordinates[1].append([event.xdata, event.ydata])
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {1}')
        else:
            self._draw_click(event)
            self.labeled_coordinates[-1].append(([event.xdata, event.ydata]))
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {-1}')

    def _on_press(self, event):
        if self._is_enter(event):
            if self.count_enter == 0:
                self.count_enter = 1
                self.which_label = 2

                self._background_figure()

            elif self.count_enter == 1:
                self.count_enter = 2
                self.figure.canvas.mpl_disconnect(self.cid1)
                self.labeled_coordinates[1] = np.array(self.labeled_coordinates[1])
                self.labeled_coordinates[-1] = np.array(self.labeled_coordinates[-1])

            elif self.count_enter == 2:
                mean1 = [np.mean(self.labeled_coordinates[1][:, 0]), np.mean(self.labeled_coordinates[1][:, 1])]
                mean2 = [np.mean(self.labeled_coordinates[-1][:, 0]), np.mean(self.labeled_coordinates[-1][:, 1])]
                self.mean_list = [mean1, mean2]

                # Under assumption that stds vary from feature to feature, but are the same regardless of y,
                # Gaussian Naive Bayes classifier has linear decision boundary.
                # So we calculate var ofself. data points for each feature dimension, only given y=1
                # And cov is zero under Naive Bayes Assumption. cov matrix here is just for stds
                self.cov = [[np.var(self.labeled_coordinates[1][:, 0]), 0], [0, np.var(self.labeled_coordinates[1][:, 1])]]

                self.y1_proba = len(self.labeled_coordinates[1]) / (
                            len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))
                self.y2_proba = len(self.labeled_coordinates[-1]) / (
                            len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))

                w3 = sum([np.log(self.y2_proba/self.y1_proba) + (self.mean_list[0][i]**2 -
                                                             self.mean_list[1][i]**2)/(2*self.cov[i][i]) for i in range(2)])
                w1 = (self.mean_list[1][0]-self.mean_list[0][0])/self.cov[0][0]
                w2 = (self.mean_list[1][1]-self.mean_list[0][1])/self.cov[1][1]

                linearGNB_weights = np.array([w1/w2, w2/w2, w3/w2])     # to prevent np.exp computation from overflowing
                self.weights = linearGNB_weights

                self._linear_Gaussian_Naive_Bayes()
                self.count_enter = 3

            else:
                self.labeled_coordinates[1] = np.c_[self.labeled_coordinates[1], np.ones(len(self.labeled_coordinates[1]))]
                self.labeled_coordinates[-1] = np.c_[self.labeled_coordinates[-1], np.ones(len(self.labeled_coordinates[-1]))]

                self.label = []
                self.X = []
                for key in self.labeled_coordinates.keys():
                    for i in range(len(self.labeled_coordinates[key])):
                        self.label.append(key)
                        self.X.append(self.labeled_coordinates[key][i])
                self.X = np.array(self.X)

                self._logistic_regression()
                self.figure.canvas.mpl_disconnect(self.cid2)

        else:
            print('You pressed a wrong button')

    def _logistic_regression(self):
        gradient = self._calculate_gradient(self.weights)
        x = np.array(range(-5, 6))
        s = np.zeros(3)

        n_iter = 0
        while n_iter < 50:
            n_iter += 1
            s = s + gradient**2
            gradient_old = gradient
            alpha = self._adagrad_learning_rate(s)
            self.weights = self.weights - np.dot(alpha, gradient)

            gradient = self._calculate_gradient(self.weights)
            loss = self._calculate_loss(self.weights)

            gradient_info = f'Gradient {n_iter}: loss {loss} norm(gradient change) {np.linalg.norm(gradient-gradient_old)}'
            print(gradient_info)
            y = [(-self.weights[0] / self.weights[1]) * x_i - self.weights[2] / self.weights[1] for x_i in x]

            plt.cla()
            self._background_figure(iter_info=gradient_info)
            self._redraw_plots()
            plt.plot(x, y)
            self._fill_btw(x, y)
            plt.pause(0.3)

            if np.linalg.norm(gradient - gradient_old) < 1e-3:
                break

        n_iter = 0
        while n_iter < 50:
            n_iter += 1
            Hessian = self._calculate_Hessian(self.weights)
            gradient_old = gradient
            self.weights = self.weights - (np.linalg.inv(Hessian) @ gradient)
            gradient = self._calculate_gradient(self.weights)

            loss = self._calculate_loss(self.weights)
            newton_info = f"Newton's method: {n_iter}: loss {loss} norm(gradient change) {np.linalg.norm(gradient-gradient_old)}"
            print(newton_info)
            y = [(-self.weights[0] / self.weights[1]) * x_i - self.weights[2] / self.weights[1] for x_i in x]

            plt.cla()
            self._background_figure(iter_info=newton_info)
            self._redraw_plots()
            plt.plot(x, y)
            self._fill_btw(x, y)
            plt.pause(0.3)

            if np.linalg.norm(gradient - gradient_old) < 1e-6:
                break

        print('Done!')
        plt.title("Converged!")

    def _calculate_gradient(self, weights):
        gradient = np.zeros(3)
        for idx, x_i in enumerate(self.X):
            z_i = self.label[idx]*np.dot(weights, x_i)
            gradient = gradient - (1-self._sigmoid(z_i))*self.label[idx]*x_i

        return gradient

    def _calculate_Hessian(self, weights):
        d = []
        for idx, x_i in enumerate(self.X):
            z_i = self.label[idx] * np.dot(weights, x_i)
            diags_ii = self._sigmoid(z_i)*(1-self._sigmoid(z_i))
            d.append(diags_ii)
        D = np.diag(d)

        Hessian = self.X.T @ D @ self.X

        return Hessian

    def _calculate_loss(self, weights):
        loss = 0
        for idx, x_i in enumerate(self.X):
            z_i = self.label[idx] * np.dot(weights, x_i)
            loss = loss - np.log(self._sigmoid(z_i))

        return loss

    def _sigmoid(self, z):
        sig_func = 1 / (1 + np.exp(-z))

        return sig_func

    def _adagrad_learning_rate(self, s):
        alpha = self.lr / np.sqrt(1e-8 + s)

        return alpha

    def _linear_Gaussian_Naive_Bayes(self):
        x = np.linspace(-5, 5, 50)
        y = [(self.cov[1][1] / (self.mean_list[1][1]-self.mean_list[0][1])) *
             (np.log(self.y1_proba/self.y2_proba) + ((self.mean_list[0][0]-self.mean_list[1][0])/self.cov[0][0])*xx +
              (self.mean_list[1][0]**2 - self.mean_list[0][0]**2)/(2*self.cov[0][0]) +
              (self.mean_list[1][1]**2 - self.mean_list[0][1]**2)/(2*self.cov[1][1])) for xx in x]

        plt.cla()
        self._background_figure()
        self._redraw_plots()
        self._show_two_Gaussians_dist()
        plt.plot(x, y)
        self._fill_btw(x, y)

    def _background_figure(self, iter_info=None):
        if self.count_enter == 0:
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('click to plot from class 1. Please press enter when finished.')

        elif self.count_enter == 1:
            self.ax.set_title('click to plot from class 2. Please double enter to start linear GNB.')

        elif self.count_enter == 2:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title(f'linear Gaussian Naive Bayes, class 1: {len(self.labeled_coordinates[1])}, '
                              f'class 2: {len(self.labeled_coordinates[-1])}')

        else:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('Logistic Regression: ' + iter_info)

    def _redraw_plots(self):
        for x, y in self.labeled_coordinates[1][:, :2]:
            self.ax.scatter(x, y, marker='.', c='r')
        for x, y in self.labeled_coordinates[-1][:, :2]:
            self.ax.scatter(x, y, marker='x', c='b')

    def _fill_btw(self, x, y):
        is_class1_above = ((self.mean_list[0][0]-self.mean_list[1][0]>0) & (self.mean_list[0][1]-self.mean_list[1][1]>0)) or \
                          ((self.mean_list[0][0]-self.mean_list[1][0]<0) & (self.mean_list[0][1]-self.mean_list[1][1]>0))

        if is_class1_above:
            plt.fill_between(x, -10, y, alpha=.25, color='b')
            plt.fill_between(x, y, 10, alpha=.25, color='r')
        else:
            plt.fill_between(x, y, 10, alpha=.25, color='b')
            plt.fill_between(x, -10, y, alpha=.25, color='r')

    def _draw_click(self, event):
        if self.which_label == 1:
            self.ax.scatter(event.xdata, event.ydata, marker='.', c='r')
        else:
            self.ax.scatter(event.xdata, event.ydata, marker='x', c='b')

    def _is_enter(self, event):
        if event.key == 'enter':
            return True
        else:
            return False

    def _show_two_Gaussians_dist(self):
        color_list = ['darkred', 'darkblue']

        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)

        # get probabilities at each point and plot contour
        for i in range(2):
            zz = np.array([scipy.stats.multivariate_normal.pdf(np.array([xx, yy])
                                                               , mean=self.mean_list[i], cov=self.cov)
                           for xx, yy in zip(np.ravel(X), np.ravel(Y))])

            Z = zz.reshape(X.shape)

            CS = plt.contour(X, Y, Z, levels=[0.005, 0.05, 0.2], alpha=.3, colors=color_list[i])
            plt.clabel(CS, inline=1, fontsize=10)