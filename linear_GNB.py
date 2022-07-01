import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class twoD_coordinates_lGNB:
    def __init__(self):
        self.count_enter = 0
        self.which_label = 1

        self._background_figure()
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        self.labeled_coordinates = {}
        self.labeled_coordinates[-1] = []
        self.labeled_coordinates[1] = []
        plt.show()

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

            else:
                mean1 = [np.mean(self.labeled_coordinates[1][:, 0]), np.mean(self.labeled_coordinates[1][:, 1])]
                mean2 = [np.mean(self.labeled_coordinates[-1][:, 0]), np.mean(self.labeled_coordinates[-1][:, 1])]
                self.mean_list = [mean1, mean2]

                # Under assumption that stds vary from feature to feature, but are the same regardless of y,
                # Gaussian Naive Bayes classifier has linear decision boundary.
                # So we calculate var ofself. data points for each feature dimension, only given y=1
                # And cov is zero under Naive Bayes Assumption. cov matrix here is just for stds
                self.cov = [[np.var(self.labeled_coordinates[1][:, 0]), 0], [0, np.var(self.labeled_coordinates[1][:, 1])]]
                self.cov2 = [[np.var(self.labeled_coordinates[-1][:, 0]), 0], [0, np.var(self.labeled_coordinates[-1][:, 1])]]
                self.cov_list = [self.cov, self.cov2]

                self.y1_proba = len(self.labeled_coordinates[1]) / (
                            len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))
                self.y2_proba = len(self.labeled_coordinates[-1]) / (
                            len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))

                self._linear_Gaussian_Naive_Bayes()
                self.figure.canvas.mpl_disconnect(self.cid2)

        else:
            print('You pressed a wrong button')

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
        plt.fill_between(x, -10, y, alpha=.25)
        plt.fill_between(x, y, 10, alpha=.25, color='r')

        print(f'mean of coordinates from class 1: {self.mean_list[0]}')
        print(f'mean of coordinates from class 2: {self.mean_list[1]}')
        print(f'variance of coordinates: {[self.cov[0][0], self.cov[1][1]]} (no differ from class 1 to class 2)')
        print(f'P(Y = class 1): {self.y1_proba}, P(Y = class 2): {self.y2_proba}')

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
            self.ax.set_title('click to plot from class 2. Please double enter to start linear GNB.')

        else:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title(f'linear Gaussian Naive Bayes, class 1: {len(self.labeled_coordinates[1])}, '
                              f'class 2: {len(self.labeled_coordinates[-1])}')

    def _redraw_plots(self):
        for x, y in self.labeled_coordinates[1]:
            self.ax.scatter(x, y, marker='.', c='r')
        for x, y in self.labeled_coordinates[-1]:
            self.ax.scatter(x, y, marker='x', c='b')

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