import nimblephysics as nimble
import numpy as np
from tqdm import tqdm


class B3DConverter:
    
    def __init__(self, geometry_path):

        self.geometry_path = geometry_path
        self.skeleton = None
        self.joint_names = []
        
    def load_subject(self, b3d_path, processing_pass=0):
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        
        self.skeleton = subject.readSkel(
            processingPass=processing_pass,
            geometryFolder=self.geometry_path
        )
        self.joint_names = [self.skeleton.getJoint(i).getName() 
                           for i in range(self.skeleton.getNumJoints())]
        
        print(f"Data joint number: {self.skeleton.getNumJoints()}")
        print(f"Data DOF number: {self.skeleton.getNumDofs()}")
        
        return subject
    
    def convert_single_trial(
        self,
        subject, 
        trial_idx, 
        processing_pass=0,
        start_frame=0,
        num_frames=None
    ):
        
        if num_frames <= 0:
            return np.array([])
        
        frames = subject.readFrames(
            trial=trial_idx,
            includeProcessingPasses=True,
            startFrame=start_frame,
            numFramesToRead=num_frames
        )
        
        #Joint position
        joint_position = []
        for frame in frames:
            positions = frame.processingPasses[processing_pass].pos
            joint_position.append(positions)
            
        return np.array(joint_position)
    
    def convert_data(
        self, 
        subject,
        processing_pass=0
    ):

        all_joint_pos = []
        
        num_trials = subject.getNumTrials()
        print(f"Extracting data of {num_trials} trial...")
        
        for trial_idx in tqdm(range(num_trials), desc="Trial data extraction"):
            trial_length = subject.getTrialLength(trial_idx)
            
            frames_to_read = trial_length
                
            joint_pos = self.convert_single_trial(
                subject=subject,
                trial_idx=trial_idx,
                processing_pass=processing_pass,
                start_frame=0,
                num_frames=frames_to_read
            )

            if len(joint_pos) > 0:
                all_joint_pos.append(joint_pos)

        if all_joint_pos:
            joint_pos_array = np.vstack(all_joint_pos)
        else:
            joint_pos_array = np.array([])

        print(f"Converting complete!")
        print(f"Total frame: {len(joint_pos_array)}")
        print(f"DOF num: {len(joint_pos_array[0]) if len(joint_pos_array) > 0 else 0}")
        
        return joint_pos_array
    
    def save_data(
        self, 
        data,
        output_path
    ):

        np.save(output_path, data)
            
        print(f"Data saved to: {output_path}")
        print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    b3d_file = "/home/kishgard/projects/Addbiomechanics/data/Dataset/Hammer2013_Formatted_With_Arm/subject01/subject01.b3d"
    output_file = "joint_positions.npy"
    geometry_path = "/home/kishgard/projects/Addbiomechanics/data/Geometry/"

    #kinematic pass is 0
    processing_pass = 0

    converter = B3DConverter(geometry_path)
    subject = converter.load_subject(b3d_file, processing_pass)
    
    joint_pos = converter.convert_data(
        subject, processing_pass
    )
    converter.save_data(
        joint_pos, output_file
    )
    