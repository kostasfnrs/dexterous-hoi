import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class SkeletonVisualizer:
    def __init__(self, skeleton_data, floor_height=0, joint_colors=None):
        self.skeleton_data = skeleton_data
        self.T = skeleton_data.shape[0]
        self.floor_height = floor_height
        if joint_colors is None:
            # self.joint_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'lime', 'olive', 'teal', 'navy', 'maroon', 'salmon', 'cyan', 'indigo', 'gold', 'peru', 'darkgreen']
            self.joint_colors = ['r'] * 22

            # right foot
            self.joint_colors[8] = 'g'
            self.joint_colors[11] = 'g'
            
            # left foot
            self.joint_colors[7] = 'b'
            self.joint_colors[10] = 'b'
        else:
            self.joint_colors = joint_colors

        # Set up the figure and axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Skeleton Animation')
        self.ax.scatter(0, 0, self.floor_height, color='gray', alpha=0.5)  # Plot floor

        for i in range(22):
            x = self.skeleton_data[frame, i, 0]
            y = self.skeleton_data[frame, i, 1]
            z = self.skeleton_data[frame, i, 2]
            self.ax.scatter(x, y, z, color=self.joint_colors[i], marker='o')

    def animate(self):
        # Create the animation
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.T, interval=50)
        # Show the animation
        plt.show()