clear;
clc;

filePath = fileparts(mfilename('fullpath'));
% Path to your repository
path2repo = [filePath filesep '..' filesep '..' filesep];

current_t = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
current_t_str = char(current_t);

% standingPath = fullfile(path2repo, 'results/IntroductionExamples/2025_09_24_script3D_standing.mat');
runningPath = fullfile(path2repo, 'results/MaroCODsims/s01/Baseline/26092025-CODinitGuess/26092025_scriptCODfromMeasIMU_CODrunning.mat');
% curvedRunningPath = fullfile('..', '..', 'results/IntroductionExamples/2025_08_23_script3D_curvedRunning.mat');

% resultStanding = load(standingPath);
% problemStanding = resultStanding.result.problem;
resultRunning = load(runningPath);
problemRunning = resultRunning.result.problem;
% resultCurvedRunning = load(curvedRunningPath);
% problemCurvedRunning = resultCurvedRunning.result.problem;


% idxStandingJointsAllNodes = problemStanding.idx.states(problemStanding.model.extractState('q'), 1:problemStanding.nNodes);
% for nodeIdx = 1:problemStanding.nNodes
%     standingJoints(:, nodeIdx) = resultStanding.result.X(idxStandingJointsAllNodes(:, nodeIdx));
% end

idxRunningJointsAllNodes = problemRunning.idx.states(problemRunning.model.extractState('q'), 1:problemRunning.nNodes);
for nodeIdx = 1:problemRunning.nNodes
    runningJoints(:, nodeIdx) = resultRunning.result.X(idxRunningJointsAllNodes(:, nodeIdx));
end

% idxCurvedRunningJointsAllNodes = problemCurvedRunning.idx.states(problemCurvedRunning.model.extractState('q'), 1:problemCurvedRunning.nNodes);
% for nodeIdx = 1:problemCurvedRunning.nNodes
%     curvedRunningJoints(:, nodeIdx) = resultCurvedRunning.result.X(idxCurvedRunningJointsAllNodes(:, nodeIdx));
% end

% filenameStanding = ['standingJoints_' current_t_str '.mat'];
filenameRunning = ['runningJoints_' current_t_str '.mat'];
% filenameCurvedRunning = ['curvedRunningJoints_' current_t_str '.mat'];

% save(filenameStanding, 'standingJoints');
save(filenameRunning, 'runningJoints');
% save(filenameCurvedRunning, 'curvedRunningJoints');
