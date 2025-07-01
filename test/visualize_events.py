import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib
matplotlib.use('TkAgg') 

def read_h5_event_components(hdf_path):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        return (f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1))
    else:
        return (f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1))

def create_animation(xs, ys, ts, ps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        start_t = frame_idx * 0.01
        end_t = start_t + 0.05
        mask = (ts >= start_t) & (ts < end_t)

        ax.cla()
        ax.set_xlim(start_t, end_t)
        ax.set_ylim(0, 346)
        ax.set_zlim(0, 260)
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
        ax.invert_zaxis() 
        ax.set_title(f"Event Visualizer")
        ax.scatter(ts[mask], xs[mask], ys[mask], c=['r' if p > 0 else 'b' for p in ps[mask]], s=1)

    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="H5 file path")
    parser.add_argument("--img_size", nargs='+', type=int, default=(260, 346))
    parser.add_argument("--max_events", type=int, default=None)
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)

    ts = ts - ts.min()

    if args.max_events:
        xs = xs[:args.max_events]
        ys = ys[:args.max_events]
        ts = ts[:args.max_events]
        ps = ps[:args.max_events]

    create_animation(xs, ys, ts, ps)
