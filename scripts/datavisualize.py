import nimblephysics as nimble
import numpy as np
import time

class PoseVisualizer:
    def __init__(self):
        self.skeleton = None
        self.gui = None

        self._load_skeleton()

    def _load_skeleton(self):
        rajagopal_opensim = nimble.RajagopalHumanBodyModel()
        self.skeleton = rajagopal_opensim.skeleton

        print("Skeleton Loaded!")

    def set_pose(self, joint_position):
        if len(joint_position) != self.skeleton.getNumDofs():
            raise ValueError("Mismatch between joint dim and skeleton DOF")
        
        self.skeleton.setPositions(joint_position)

    def visualize_pose(self, joint_position, port=8080):
        self.set_pose(joint_position)

        self.gui = nimble.NimbleGUI()
        self.gui.serve(port)

        while True:
            self.gui.nativeAPI().renderSkeleton(self.skeleton)
            time.sleep

    def animate_poses(self, joint_positions, port=8080):
        self.gui = nimble.NimbleGUI()
        self.gui.serve(port)
        frame_time = 1.0/30.0

        while True:
            for i, positions in enumerate(joint_positions):
                start_time = time.time()

                self.set_pose(positions)
                self.gui.nativeAPI().renderSkeleton(self.skeleton)

                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

if __name__ == "__main__":
    data_file = "joint_positions.npy"

    joint_pos = np.load(data_file)
    visualizer = PoseVisualizer()

    visualizer.animate_poses(joint_positions=joint_pos)
