function output = vaeReconstructionTerm(obj, option, X, vaeParams)

fctname = 'vaeReconstructionTerm';

if strcmp(option,'init')
    
    if ~isfield(obj.idx,'states')
        error('Model states are not stored in state vector X.')
    end
    
    
    obj.objectiveInit.(fctname).idxJointsAllNodes = obj.idx.states(obj.model.extractState('q'), 1:obj.nNodes);
    obj.objectiveInit.(fctname).vaeParams = vaeParams;

    % %Indices for knee joint after removing pelvis (for flip)
    % obj.objectiveInit.(fctname).kneeIndices = [4, 11];
    
    try
        py.sys.path().append(vaeParams.pythonPath);
        obj.objectiveInit.(fctname).vaeModule = py.importlib.import_module('vaemodel');
        
        success = obj.objectiveInit.(fctname).vaeModule.initialize_vae(...
            vaeParams.modelPath, ...
            vaeParams.scalerPath, ...
            pyargs('num_dofs', int32(vaeParams.numDofs), ...
                   'latent_dim', int32(vaeParams.latentDim), ...
                   'hidden_dim', int32(vaeParams.hiddenDim), ...
                   'device', vaeParams.device));
        
        if ~success
            error('Failed to initialize VAE model');
        end
        
    catch ME
        error('Failed to initialize Python VAE interface: %s', ME.message);
    end
    
    obj.objectiveInit.(fctname).nJoints = length(obj.model.extractState('q'));
    
    output = NaN;
    return;
end

idxJointsAllNodes = obj.objectiveInit.(fctname).idxJointsAllNodes;
vaeModule = obj.objectiveInit.(fctname).vaeModule;
% kneeIndices = obj.objectiveInit.(fctname).kneeIndices;

if strcmp(option,'objval')
    totalCost = 0;
    
    for nodeIdx = 1:obj.nNodes
        jointIndices = idxJointsAllNodes(7:33, nodeIdx);
        currentJoints = X(jointIndices);
        jointsForVAE = currentJoints;
        % jointsForVAE(kneeIndices) = -jointsForVAE(kneeIndices);

        try
            pyJoints = py.numpy.array(jointsForVAE);
            pyResult = vaeModule.reconstruct(pyJoints);
            nodeError = double(pyResult);
            totalCost = totalCost + nodeError;
            
        catch ME
            error('VAE reconstruction failed for node %d: %s', nodeIdx, ME.message);
        end
    end
    
    output = totalCost / obj.nNodes;
    
elseif strcmp(option,'gradient')
    output = zeros(size(X));
    
    for nodeIdx = 1:obj.nNodes
        jointIndices = idxJointsAllNodes(7:33, nodeIdx);
        currentJoints = X(jointIndices);
        jointsForVAE = currentJoints;
        % jointsForVAE(kneeIndices) = -jointsForVAE(kneeIndices);
        
        try
            pyJoints = py.numpy.array(jointsForVAE);
            pyResult = vaeModule.reconstruct_withgrad(pyJoints);
            gradient = double(pyResult);
            gradient(kneeIndices) = -gradient(kneeIndices);
            
            output(jointIndices) = gradient;
            
        catch ME
            error('Analytical gradient computation failed for node %d: %s', nodeIdx, ME.message);
        end
    end
    
    output = output / obj.nNodes;
end

end