import matplotlib.pyplot as plt
import numpy as np

class twoD_coordinates_Perceptron:
    def __init__(self, learning_rate=1, n_iters=30):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.count_enter = 0
        self.which_label = 0

        self._background_figure()
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        self.coordinates = []
        self.label = []
        self.weights_record = []
        plt.show()

    def labeled_coordinates(self):
        return self.coordinates, self.label

    def weights_ubdate_list(self):
        return self.weights_record

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

            elif self.count_enter == 1:
                self.count_enter = 2
                self.figure.canvas.mpl_disconnect(self.cid1)

                self.X = np.c_[self.coordinates, np.ones(len(self.coordinates))]     # one more dimension with 1 to take b
                self.weights = np.zeros(3)                                           # one more dimension to take b

                self.n_iter = 1
                self.m = 1
                self.x = np.array(range(-5, 6))
            else:
                if self.m > 0:
                    self._Perceptron()
                    self.n_iter += 1
                else:
                    self.ax.set_title('Converged!')
        else:
            print('You pressed a wrong button')
    def _Perceptron(self):
        self.m = 0
        if self.n_iter < self.n_iters:
            for idx, x_i in enumerate(self.X):
                if np.dot(x_i, self.weights) * self.label[idx] <= 0:
                    self.weights = self.weights + self.lr * self.label[idx] * x_i
                    self.weights_record.append(self.weights)
                    self.b = self.weights[2]
                    self.y = [-(self.weights[0] / self.weights[1]) * x_i - (1 / self.weights[1]) * self.weights[2] for x_i in self.x]    # y = w^T*x hypherplane
                    self.m += 1

            plt.cla()
            self._background_figure()
            self._redraw_plots()
            plt.plot(self.x, self.y)
            plt.fill_between(self.x, -10, self.y, alpha=.25)
            plt.fill_between(self.x, self.y, 10, alpha=.25, color='r')
        else:
            self.ax.set_title('Screwed. It is probably not linearly seperable.')
            self.figure.canvas.mpl_disconnect(self.cid2)

    def _background_figure(self):
        if self.count_enter == 0:
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('click to plot from class 1. Please press enter when finished.')
        elif self.count_enter == 1:
            self.ax.set_title('click to plot from class 2. please double enter to start Perceptron.')
        else:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title(f'iteration: {self.n_iter}, max_iteration: {self.n_iters}, mistakes: {self.m}, n_points:{len(self.label)}, stepsize: {self.lr}')

    def _redraw_plots(self):
        for idx, y_i in enumerate(self.label):
            if y_i == 1:
                self.ax.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='.', c='r')
            else:
                self.ax.scatter(self.coordinates[idx][0], self.coordinates[idx][1], marker='x', c='b')

    def _draw_click(self, event):
        if self.which_label == 0:
            self.ax.scatter(event.xdata, event.ydata, marker='.', c='r')
        else:
            self.ax.scatter(event.xdata, event.ydata, marker='x', c='b')

    def _unit_step_func(self, dot_product):
        return np.where(dot_product<=0, -1, 1)

    def _is_enter(self, event):
        if event.key == 'enter':
            return True
        else:
            return False