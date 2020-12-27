import matplotlib.pyplot as plt
import time
import sys
import math
from matplotlib.widgets import Slider, Button
import matplotlib.animation as anim
import numpy as np


class column:

    def __cache_states(self, states):
        """Create a chache of states"""

        for s in range(int(states/self.step_size)):
            for k in range(len(self.k)):
                self.states[k].append(self.plates[k])
            for x in range(self.step_size):
                self.step()
                print(round((s * self.step_size + x) / (states) * 100, 1), "% Complete")
        self.plates = self.initial
        self.state = 0

    def __init__(self, n, k=[1, 2], step_size=5):
        assert step_size < n
        assert n >= 10
        assert step_size > 0
        self.n = n
        self.k = k
        self.step_size = step_size
        self.plates = [[] for x in self.k]
        self.state = 0
        self.steps = int(max(k) + 3) * n
        for x in range(n):
            for y in range(len(k)):
                self.plates[y].append((0, 0))  # index 0: mobile, index 1: stationary
        for i in range(int(n/10)):
            for p in self.plates:
                p[i] = (1.0, 0)
        self.initial = list.copy(self.plates)
        self.states = [[] for x in self.k]
        sys.setrecursionlimit(10000)
        start = time.time()
        self.__cache_states(self.steps)
        end = time.time()
        print("Calculated", self.steps, "states in", round(end-start, 2), "seconds.")

    def __equilibrate(self, tuple, k):
        """Equilibrate a plate given a tuple of (mobile, stationary) and a k value"""

        total_conc = tuple[0] + tuple[1]
        mobile = 1 / (1 + k) * total_conc
        stationary = total_conc - mobile
        return (mobile, stationary)

    def __recursive_move_mobile(self, plates):
        """Recursive method to move the mobile phase down one plate.
        Much slower than iterative
        """
        current = plates[-1:][0]  # The last entry in the list of plates
        if(len(plates) == 1):
            return [(0, current[1])]
        previous = plates[-2:-1][0]  # The second-to-last entry in the list of plates
        new_mobile = previous[0]
        stationary = current[1]
        return self.__recursive_move_mobile(plates[:-1]) + [(new_mobile, stationary)]

    def __iterative_move_mobile(self, plates):
        """Iterative method to move the mobile phase down one plate"""

        mobiles = []
        stationaries = []
        for p in plates:
            mobiles.append(p[0])
            stationaries.append(p[1])
        new_mobiles = [0] + mobiles[:-1]
        new_plates = []
        for p in range(self.n):
            new_plates.append((new_mobiles[p], stationaries[p]))
        return new_plates

    def print_plates(self):
        """Prints all the plates vertically"""

        for k in range(len(self.k)):
            print()
            print('Compound', k)
            for p in self.plates[k]:
                print(p)

    def set_plate(self, index, tuple):
        """This method allows you to manually specify the values at a plate"""
        for i in tuple:
            assert i <= 1
            assert i >= 0
        self.plates[index] = tuple

    def step(self):
        """Perform one step. A step involves moving all the mobile phases
        down one plate and then re-equilibrating all the plates
        """
        for k in range(len(self.k)):
            self.plates[k] = self.__iterative_move_mobile(self.plates[k])
            for i in range(len(self.plates[0])):
                self.plates[k][i] = self.__equilibrate(self.plates[k][i], self.k[k])
        self.state += 1

    def recalculate_cache(self, cached_steps):
        """Recalculate the step caches to the given number of cached steps"""

        self.__cache_states(cached_steps)

    def go_to_step(self, step):
        """Set the column to the state at the given step
        If not cached, calculates that step, can take a while
        """
        self.state = step
        self.plates = self.initial
        if step < len(self.states[0]):
            for k in range(len(self.k)):
                self.plates[k] = self.states[k][int(step)]
        else:
            for x in range(int(step)):
                self.step()

    def combine_compounds(self):
        """All compounds are calculated indepedendently.
        This method combines the values for each compound to plot
        one one axis
        """
        combined = [0 for x in range(len(self.plates[0]))]
        for k in range(len(self.k)):
            for i in range(len(self.plates[0])):
                combined[i] += self.plates[k][i][0]
                combined[i] += self.plates[k][i][1]
        return combined

    def plot(self):
        """Plot the column with a slider"""
        Y = self.combine_compounds()
        fig, ax = plt.subplots()
        p, = plt.plot(Y)
        plt.ylim(0, len(self.k) / 1.5)

        plt.subplots_adjust(left=0.25, bottom=0.25)
        axcolor = 'lightgoldenrodyellow'
        axstep = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        s_step = Slider(axstep, 'Step', 0, int((self.steps-1) /
                        self.step_size), valinit=0, valstep=1)

        def update(val):
            self.go_to_step(val)
            nonlocal Y
            Y = self.combine_compounds()
            p.set_ydata(Y)
            fig.canvas.draw_idle()

        s_step.on_changed(update)
        plt.show()
    
    def save_movie(self):
        """Save the graph as .png files"""
        for s in range(len(self.states[0])):
            self.go_to_step(s)
            Y = self.combine_compounds()
            plt.plot(Y)
            plt.ylim(0, len(self.k) / 1.5)
            plt.savefig(str(s).zfill(3) + '.png', dpi=300)
            #plt.clf()


if __name__ == "__main__":
    c = column(1000, [0.2, 0.3, 0.5, 0.7, 1.1, 1.4, 1.7] , 4)
    #c.plot()
    c.save_movie()
