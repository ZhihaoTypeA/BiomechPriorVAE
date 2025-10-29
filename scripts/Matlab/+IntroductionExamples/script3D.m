%======================================================================
%> @file IntroductionExamples/script3D.m
%> @brief Script to start working with the Bio-Sim-Toolbox for 3D models
%> @example
%> @details
%> This is a script which you can use if you work with our code for the
%> first time. The settings are based on the following paper:
%> Nitschke, M., Dorschky, E., Heinrich, D., Schlarb, H., Eskofier, B. M.,
%> Koelewijn, A. D., & van den Bogert, A. J. (2020). Efficient trajectory
%> optimization for curved running using a 3D musculoskeletal model with
%> implicit dynamics. Scientific reports, 10(1), 17655.
%>
%> @author Marlies Nitschke, Anne Koelewijn
%> @date September, 2024
%======================================================================


clear all 
close all
clc

%% Settings
% Get path of this script
filePath = fileparts(mfilename('fullpath'));
% Path to your repository
path2repo = [filePath filesep '..' filesep '..' filesep];


% Fixed settings
dataFolder     = 'data/IntroductionExamples';    % Relative from the path of the repository
dataFile       = 'running_3D.mat';               % Straight running data used in 2020 paper
dataFileCurved = 'curvedRunning_3D.mat';         % Curved running data used in 2020 paper
modelFile      = 'gait3d_pelvis213.osim';        % Name of our default model. You can add a new model in the osim_files folder of Gait3d, but you can also specify the path to a model similar to how it is done for the dataFile
resultFolder   = 'results/IntroductionExamples'; % Relative from the path of the repository

% VAE Configuration
%enable flag
useVAE = false;

if useVAE
    vaeParams.modelPath = fullfile(filePath, 'BiomechPriorVAE_best.pth');
    vaeParams.scalerPath = fullfile(filePath, 'scaler.pkl');
    vaeParams.pythonPath = filePath;
    
    vaeParams.numDofs = 27; %excluding pelvis
    vaeParams.latentDim = 20;
    vaeParams.hiddenDim = 512;
    vaeParams.device = 'cpu';
    vaeParams.weight = 25;

    fprintf('VAE constraints enabled\n')
else
    fprintf('VAE constraints disabled\n')
end


%% Initalization
% Get date
dateString = datestr(date, 'yyyy_mm_dd');

% Get absolute file names
resultFileStanding      = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_standing'];
resultFileRunning       = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_running'];
resultFileCurvedRunning = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_curvedRunning'];
dataFile                = [path2repo,filesep,dataFolder,  filesep,dataFile];
dataFileCurved          = [path2repo,filesep,dataFolder,  filesep,dataFileCurved];

% Create resultfolder if it does not exist
if ~exist([path2repo,filesep,resultFolder], 'dir')
    mkdir([path2repo,filesep,resultFolder]);
end


%% Standing: Simulate standing with minimal effort without tracking data for one point in time (static)
% Create an instane of our 3D model class using the default settings
model = Gait3d(modelFile);
% => We use a mex function for some functionality of the model. This was
% automatically initialized with the correct settings for the current
% model. (see command line output)

if useVAE
    standingVAE = vaeParams;
    % standingVAE.weight = 1e-3;
    % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    % => Have a look into it ;)
    problemStanding = IntroductionExamples.standing3D(model, resultFileStanding, standingVAE);
else
    problemStanding = IntroductionExamples.standing3D(model, resultFileStanding);
end

% Create an object of class solver. We use most of the time the IPOPT here.
solver = IPOPT();

% Change settings of the solver
solver.setOptionField('tol', 1e-4); % 0.0000001
solver.setOptionField('dual_inf_tol', 1e-4);
solver.setOptionField('acceptable_tol', 1e-4); 

% Solve the optimization problem
resultStanding = solver.solve(problemStanding);

% Save the result
resultStanding.save(resultFileStanding);

% To plot the result we have to extract the states x from the result vector X
x = resultStanding.X(resultStanding.problem.idx.states);

% Now, we can plot the stick figure visualizing the result
standingFig = figure();
resultStanding.problem.model.showStick(x);
title('3D Standing'); 
current_t = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
current_t_str = char(current_t);
if useVAE
    filenameFig = ['standing_VAE_' current_t_str '.png'];
else
    filenameFig = ['standing_' current_t_str '.png'];
end

try
    exportgraphics(standingFig, filenameFig);
catch
    saveas(standingFig, filenameFig);
end

% If the model is standing on the toes, this is a local optimum. Rerun this
% section and you will find a different solution, due to a different random
% initial guess.

%% Straight running: simulate running with minimal effort while tracking data
% Simulation settings.
N = 50; % number of collocation nodes
sym = 1; % do not simulate symmetric movement
W.effMuscles = 0; % Weight of effort term in objective 1000
W.effMusclesAct = 1.0e3; %2.0e3 / 92;
W.effMusclesTor = 1; %1.0e3 / 15;
W.cot = 0; % 5.0e2 / 92 / 74.9646;
W.effTorques = 0;    % Weight of torque term in objective
W.reg        = 0; % Weight of regularization term in objective
W.track      = 0;    % Weight of tracking term in objective
W.dur        = 0;    % No predefined duration
W.speed      = Inf;  % Predefined speed
initialGuess = resultFileStanding;

% Load and resample tracking data 
trackingData = TrackingData.loadStruct(dataFile);
trackingData.preprocessData(N);
% Extract speed and duration
targetspeed_x =  trackingData.variables.mean{strcmp(trackingData.variables.type,'speed') & strcmp(trackingData.variables.name,'x')}; % speed in x direction
targetspeed_z =  trackingData.variables.mean{strcmp(trackingData.variables.type,'speed') & strcmp(trackingData.variables.name,'z')}; % speed in z direction
targetdur =  trackingData.variables.mean{strcmp(trackingData.variables.type,'duration')}; % duration

fprintf('targetspeed_x: %f', targetspeed_x);
fprintf('targetspeed_z: %f', targetspeed_z);
fprintf('targetdur: %f', targetdur);

% Create and automatically initalize an instance of our 2D model class.  
% To fit the tracking data we have to scale the default model. This is done
% using the height and mass of the participant.
model = Gait3d(modelFile);

% Call IntroductionExamples.running3D() to specify the optimizaton problem
% => Take a look inside the function ;)
if useVAE
    straightRunningVAE = vaeParams;
    % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    % => Have a look into it ;)
    problemRunning = IntroductionExamples.running3D(model,trackingData,initialGuess,resultFileRunning,N,sym,W, targetspeed_x, targetspeed_z, targetdur, straightRunningVAE);
else
    problemRunning = IntroductionExamples.running3D(model,trackingData,initialGuess,resultFileRunning,N,sym,W, targetspeed_x, targetspeed_z, targetdur);
end


% Create solver and change solver settings
solver = IPOPT();
solver.setOptionField('max_iter', 5000);
solver.setOptionField('tol', 1e-3); % 0.0005
solver.setOptionField('dual_inf_tol', 1e-3);
solver.setOptionField('acceptable_tol', 1e-3); 


% Solve the optimization problem and save the result. 
resultRunning = solver.solve(problemRunning);
resultRunning.save(resultFileRunning); 

%% Curved running: simulate running with minimal effort while tracking data
% Curved running: Load curved data to extract desired speed and duration
speedDurCurve = TrackingData.loadStruct(dataFileCurved);
% speedDurCurve.resampleData(N);
targetspeed_x_curve =  speedDurCurve.variables.mean{strcmp(speedDurCurve.variables.type,'speed') & strcmp(speedDurCurve.variables.name,'x')}; % speed in x direction
targetspeed_z_curve =  speedDurCurve.variables.mean{strcmp(speedDurCurve.variables.type,'speed') & strcmp(speedDurCurve.variables.name,'z')}; % speed in z direction
targetdur_curve =  speedDurCurve.variables.mean{strcmp(speedDurCurve.variables.type,'duration')}; % duration

% Simulation settings. We can use the same model as before.
N = 50; % number of collocation nodes
sym = 1; % do not simulate symmetric movement
Euler = 'BE'; % 'BE' Backward Euler or 'ME' Midpoint Euler discretization
W.effMuscles = 0; % Weight of effort term  in objective, both effort terms are increased with a factor 10 because we are now making a prediction
W.effTorques = 0;    % Weight of torque term  in objective
W.effMusclesAct = 1.0e3; %2.0e3 / 92;
W.effMusclesTor = 1; %1.0e3 / 15;
W.cot = 0; % 5.0e2 / 92 / 74.9646;
W.reg        = 0;  % Weight of regularization term in objective
W.track      = 0;     % Weight of tracking term in objective
W.dur        = 0;     % No predefined duration
W.speed      = Inf;   % Predefined speed

% Call IntroductionExamples.running3D() to specify the optimizaton problem
if useVAE
    curvedRunningVAE = vaeParams;
    % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    % => Have a look into it ;)
    problemCurvedRunning = IntroductionExamples.running3D(model,trackingData,initialGuess,resultFileCurvedRunning,N,sym,W, targetspeed_x_curve, targetspeed_z_curve, targetdur_curve, curvedRunningVAE);
else
    problemCurvedRunning = IntroductionExamples.running3D(model,trackingData,initialGuess,resultFileCurvedRunning,N,sym,W, targetspeed_x_curve, targetspeed_z_curve, targetdur_curve);
end


% Create solver and change solver settings
solver = IPOPT();
solver.setOptionField('max_iter', 5000);
solver.setOptionField('tol', 1e-3); % 0.0005
solver.setOptionField('dual_inf_tol', 1e-3);
solver.setOptionField('acceptable_tol', 1e-3); 

% Solve the optimization problem and save the result. 
resultCurvedRunning = solver.solve(problemCurvedRunning);
resultCurvedRunning.save(resultFileCurvedRunning);

%% Create plots showing all simulation results at the same time

% Plot tracking data
% The following command makes all plotted figures appear in one window:
set(0, 'DefaultFigureWindowStyle', 'docked');
% And switch off the warnings this produces
warning('off', 'MATLAB:Figure:SetOuterPosition');

% You can plot all entries of the trackingData object using
plotVarTable(trackingData.variables);

% Define what should be included into the report and which simulated
% variables you want to extract first. See the documentation of
% extractData() for details.
% You can for example not extract neural excitation by using an empty cell
settings.u = {};
% If you want to extract something what had an empty cell before, you
% simply have to define in a cell what you want to extract. E.g. by using
% the names of the right leg muscles.
settings.muscleForce = resultRunning.problem.model.muscles.Properties.RowNames(1:43)';

simVarTables = cell(1, numel(2));
simVarTables{1} = resultRunning.problem.extractData(resultRunning.X,settings);
simVarTables{2} = resultCurvedRunning.problem.extractData(resultCurvedRunning.X,settings);    

% We can use a cell vector or cell matrix of simVarTables as input.
Collocation.plotMultSimVarTables(simVarTables);


%% Result extraction
idxStandingJointsAllNodes = problemStanding.idx.states(problemStanding.model.extractState('q'), 1:problemStanding.nNodes);
for nodeIdx = 1:problemStanding.nNodes
    standingJoints(:, nodeIdx) = resultStanding.X(idxStandingJointsAllNodes(:, nodeIdx));
end

idxRunningJointsAllNodes = problemRunning.idx.states(problemRunning.model.extractState('q'), 1:problemRunning.nNodes);
for nodeIdx = 1:problemRunning.nNodes
    runningJoints(:, nodeIdx) = resultRunning.X(idxRunningJointsAllNodes(:, nodeIdx));
end

idxCurvedRunningJointsAllNodes = problemCurvedRunning.idx.states(problemCurvedRunning.model.extractState('q'), 1:problemCurvedRunning.nNodes);
for nodeIdx = 1:problemCurvedRunning.nNodes
    curvedRunningJoints(:, nodeIdx) = resultCurvedRunning.X(idxCurvedRunningJointsAllNodes(:, nodeIdx));
end

if useVAE
    filenameStanding = ['standingJoints_VAE_' current_t_str '.mat'];
    filenameRunning = ['runningJoints_VAE_' current_t_str '.mat'];
    filenameCurvedRunning = ['curvedRunningJoints_VAE_' current_t_str '.mat'];
else
    filenameStanding = ['standingJoints_' current_t_str '.mat'];
    filenameRunning = ['runningJoints_' current_t_str '.mat'];
    filenameCurvedRunning = ['curvedRunningJoints_' current_t_str '.mat'];
end

save(filenameStanding, 'standingJoints');
save(filenameRunning, 'runningJoints');
save(filenameCurvedRunning, 'curvedRunningJoints');