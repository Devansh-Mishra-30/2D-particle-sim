import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class ParticleVisualizer:
    def __init__(self, filename):
        self.frames = []
        self.radii = []
        current_frame = []
        current_radii = []
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("F "):  # 帧头识别
                    if current_frame:
                        self.frames.append(np.array(current_frame))
                        self.radii.append(np.array(current_radii))
                        current_frame = []
                        current_radii = []
                else:
                    try:
                        x, y, r = map(float, line.strip().split())
                        current_frame.append([x, y])
                        current_radii.append(r)
                    except:
                        pass
            if current_frame:
                self.frames.append(np.array(current_frame))
                self.radii.append(np.array(current_radii))
        
        # 初始化图形参数
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(-250, 250)
        self.ax.set_ylim(-250, 250)  # 保持与SFML坐标系一致
        self.scat = self.ax.scatter([], [], s=100, c='blue', alpha=0.6)
        
        # 绘制边界
        boundary = plt.Circle((0, 0), 20, color='r', fill=False, linewidth=1)
        self.ax.add_patch(boundary)
        self.ax.set_aspect('equal')
    
    def update(self, frame):
        if frame < len(self.frames):
            self.scat.set_offsets(self.frames[frame])
            
            sizes = 2 * (np.array(self.radii[frame]) ** 2)  # s参数是面积，所以用直径平方
            self.scat.set_sizes(sizes)
        return self.scat,
    
    def animate(self):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.frames),
            interval=50, blit=True, repeat=False
        )
        plt.show()

if __name__ == "__main__":
    print("Loading simulation data...")
    vis = ParticleVisualizer("particles.txt")
    print(f"Loaded {len(vis.frames)} frames")
    vis.animate()