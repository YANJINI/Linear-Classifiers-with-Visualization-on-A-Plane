import matplotlib.pyplot as plt
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
plt.style.use('ggplot')

class twoD_softmarginSVM:
    def __init__(self, C):
        self.C = C

        if self.C == 0.:
            raise ValueError('C=0 makes it hard magin SVM which jams cvxopt.qp')

        self.count_enter = 0
        self.which_label = 0
        self.coordinates = []
        self.label = []
        self._background_figure()

        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        plt.show()

    def labeled_coordinates(self):
        return self.coordinates, self.label

    def alphas_w_b(self):
        return self.alphas, self.w, self.b

    def with_different_C(self):
        self.count_enter = 3

        fig, axes = plt.subplots(2, 4, figsize=(18, 10))

        self.C = 0.0001

        for i in range(2):
            for j in range(4):
                self._background_figure(ax_diffC=axes[i, j])
                self._redraw_plots()
                self._softmarginSVM()
                self.C *= 10

    def _on_click(self, event):
        if self.which_label == 0:
            self._draw_click(event)
            self.coordinates.append([event.xdata, event.ydata])
            self.label.append(1)
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {1}')
        else:
            self._draw_click(event)
            self.coordinates.append(([event.xdata, event.ydata]))
            self.label.append(-1)
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {-1}')

    def _on_press(self, event):
        if self._is_enter(event):
            if self.count_enter == 0:
                self.count_enter = 1
                self.which_label = 1
                self._background_figure()

            else:
                self.count_enter = 2
                self.figure.canvas.mpl_disconnect(self.cid1)
                self.figure.canvas.mpl_disconnect(self.cid2)
                plt.cla()
                self._background_figure()
                self._redraw_plots()
                self._softmarginSVM()

        else:
            print('You pressed a wrong button')

    def _softmarginSVM(self):
        n = len(self.coordinates)
        X = np.array(self.coordinates)
        y = np.array(self.label).reshape(-1, 1) * 1.0
        H = (y*X)@((y*X).T)                     # H_ij = y_i * y_j * dot(x_i, x_j)
        q = np.repeat([-1.0], n)                # a vector with all entries of -1
        A = y.reshape(1, -1)
        b = 0.0

        if self.C == 0:
            G = np.eye(n) * -1
            h = np.zeros(n)
        else:
            G = np.vstack((np.eye(n)*-1, np.eye(n)))
            h = np.hstack((np.zeros(n), np.repeat([self.C], n))).reshape(-1, 1)

        P = matrix(H)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        sol = solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol['x'])

        self.w = ((y*self.alphas).T@X).reshape(-1, 1)
        S = (self.alphas > 1e-4).flatten()
        self.b = np.mean(y[S] - np.dot(X[S], self.w))

        print(f'w vector: {self.w.reshape(1, -1)}\nb: {self.b}')

        x_1 = np.linspace(-5, 5, 20)
        x_2 = [-(self.w[0, 0] / self.w[1, 0]) * xx - (1 / self.w[1, 0]) * self.b for xx in x_1]
        x_3 = [-(self.w[0, 0] / self.w[1, 0]) * xx + (1 / self.w[1, 0]) * (1-self.b) for xx in x_1]
        x_4 = [-(self.w[0, 0] / self.w[1, 0]) * xx + (1 / self.w[1, 0]) * (-1-self.b) for xx in x_1]

        if self.count_enter < 3:
            plt.plot(x_1, x_2, color='k')
            plt.plot(x_1, x_3, color='w')
            plt.plot(x_1, x_4, color='w')
            plt.fill_between(x_1, -10, x_2, alpha=.25, color='b')
            plt.fill_between(x_1, x_2, 10, alpha=.25, color='r')
        else:
            self.ax_diffC.plot(x_1, x_2, color='k')
            self.ax_diffC.plot(x_1, x_3, color='w')
            self.ax_diffC.plot(x_1, x_4, color='w')
            self.ax_diffC.fill_between(x_1, -10, x_2, alpha=.25, color='b')
            self.ax_diffC.fill_between(x_1, x_2, 10, alpha=.25, color='r')

    def _background_figure(self, ax_diffC=None):
        if self.count_enter == 0:
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('click to plot from class 1. Please press enter when finished.')
        elif self.count_enter == 1:
            self.ax.set_title('click to plot from class 2. please enter to start Soft Margin SVM.')
        elif self.count_enter == 2:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title(f'Soft Margin SVM (C={self.C})')
        else:
            self.ax_diffC = ax_diffC
            self.ax_diffC.set_aspect(1)
            self.ax_diffC.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax_diffC.set_aspect('equal')
            self.ax_diffC.set_xlabel('feature 1')
            self.ax_diffC.set_ylabel('feature 2')
            self.ax_diffC.set_title(f'C={self.C}')

    def _redraw_plots(self):
        if self.count_enter < 3:
            for idx, y_i in enumerate(self.label):
                if y_i == 1:
                    self.ax.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='.', c='r')
                else:
                    self.ax.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='x', c='b')
        else:
            for idx, y_i in enumerate(self.label):
                if y_i == 1:
                    self.ax_diffC.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='.', c='r')
                else:
                    self.ax_diffC.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='x', c='b')

    def _draw_click(self, event):
        if self.which_label == 0:
            self.ax.scatter(event.xdata, event.ydata, marker='.', c='r')
        else:
            self.ax.scatter(event.xdata, event.ydata, marker='x', c='b')

    def _is_enter(self, event):
        if event.key == 'enter':
            return True
        else:
            return False