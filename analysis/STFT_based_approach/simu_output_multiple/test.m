%% CONFIGURATION
clear; clc; close all;


CSV_FILE = '/Users/noedaniel/Desktop/InterNest/Analyzer/birds_loc/simu_output_multiple/tdoa_envelope_detections.csv';

% Positions des 8 micros
micros = [
    0, 0, 0;      100, 0, 0;    0, 100, 0;    100, 100, 0;
    0, 0, 40;     100, 0, 40;   0, 100, 40;   100, 100, 40
];

% Paramètres
refMicIdx = 1;
tarMicIndices = 2:8; 
numPairs = length(tarMicIndices); % 7 Paires

%% 1. CONFIGURATION DU FUSER (Triangulation)
fuser = staticDetectionFuser(...
    'MaxNumSensors', numPairs, ... 
    'MeasurementFormat', 'custom', ...
    'MeasurementFusionFcn', @myHelperTDOA2Pos, ...
    'MeasurementFcn', @myHelperMeasureTDOA, ...
    'DetectionProbability', 0.9, ...
    'FalseAlarmRate', 1e-5, ...
    'Volume', 1e6, ...
    'UseParallel', false);

%% 2. CONFIGURATION DU TRACKER (Suivi Temporel)
tracker = trackerGNN(...
    'FilterInitializationFcn', @initKalmanFilter, ...
    'AssignmentThreshold', 20, ... 
    'DeletionThreshold', [3 5], ... 
    'ConfirmationThreshold', [2 3]); 

%% 3. CHARGEMENT ET INITIALISATIONS
if ~isfile(CSV_FILE), error('CSV introuvable !'); end
T = readtable(CSV_FILE);
uniqueTimes = unique(T.Time);
fprintf('Chargé : %d détections TDOA.\n', height(T));

warning('off', 'all'); % On coupe les warnings pour l'init

% --- A. INIT FUSER ---
fprintf('Initialisation Fuser...\n');
initDets = {};
for i = 1:numPairs
    pairInfo = struct('MicRefPos', micros(refMicIdx,:), 'MicTarPos', micros(tarMicIndices(i),:));
    initDets{end+1} = objectDetection(0, 0, ...
        'MeasurementNoise', 1, 'SensorIndex', i, 'MeasurementParameters', pairInfo);
end
[~,~] = fuser(initDets); 
reset(fuser);

% --- B. INIT TRACKER ---
fprintf('Initialisation Tracker...\n');
dummyPosDet = objectDetection(0, [50; 50; 20], 'MeasurementNoise', eye(3));

% --- CORRECTION ICI : On met des {} pour passer une liste ---
tracker({dummyPosDet}, 0); 
% -----------------------------------------------------------
reset(tracker);

warning('on', 'all'); 
fprintf('Pipeline opérationnel.\n');

%% 4. BOUCLE DE TRACKING
allTracksHistory = []; 

figure('Color','w'); hold on; grid on; axis equal; view(3);
plot3(micros(:,1), micros(:,2), micros(:,3), 'k^', 'MarkerFaceColor', 'k');
hBirds = scatter3(NaN, NaN, NaN, 80, 'filled');
colors = lines(10);
xlim([0 100]); ylim([0 100]); zlim([0 40]);
title('Tracking Multi-Oiseaux (Fuser + Tracker)');

fprintf('Lancement sur %d frames...\n', length(uniqueTimes));

for k = 1:length(uniqueTimes)
    t_curr = uniqueTimes(k);
    currentRows = T(T.Time == t_curr, :);
    
    formattedDets = {};
    
    % --- Préparation TDOA ---
    for i = 1:height(currentRows)
        row = currentRows(i, :);
        if row.MicRef ~= 1, continue; end
        
        pairIdx = row.MicTar - 1; 
        meas = row.Delay;
        noiseVar = 1e-5 / (row.Score^2); 
        
        pairInfo = struct('MicRefPos', micros(1,:), 'MicTarPos', micros(row.MicTar,:));
        
        det = objectDetection(t_curr, meas, ...
            'MeasurementNoise', noiseVar, ...
            'SensorIndex', pairIdx, ...
            'MeasurementParameters', pairInfo);
            
        formattedDets{end+1} = det;
    end
    
    
    % --- Étape 1 : Fusion (TDOA -> Positions) ---
    fusedDets = {};
    
    % On vérifie d'abord combien de PAIRES UNIQUES ont détecté quelque chose
    if ~isempty(formattedDets)
        % Extraction des SensorIndex de toutes les détections
        sensorIndices = cellfun(@(d) d.SensorIndex, formattedDets);
        numUniqueSensors = length(unique(sensorIndices));
    else
        numUniqueSensors = 0;
    end
    
    % CONDITION STRICTE : 
    % 1. Au moins 3 mesures au total (pour la stabilité mathématique)
    % 2. Au moins 2 paires différentes (pour avoir une intersection géométrique)
    if length(formattedDets) >= 3 && numUniqueSensors >= 2
        try
            [fusedDets, ~] = fuser(formattedDets);
        catch ME
            % Si le fuser plante quand même (ex: géométrie dégénérée), on ignore cette frame
            warning('Echec fusion à T=%.2f : %s', t_curr, ME.message);
        end
    end

    
    % --- Étape 2 : Tracking ---
    confirmedTracks = [];
    if ~isempty(fusedDets)
        confirmedTracks = tracker(fusedDets, t_curr);
    else
        % Maintenant que le tracker est bien init avec une {}, ceci passe !
        confirmedTracks = tracker({}, t_curr);
    end
    
    % --- Affichage ---
    if ~isempty(confirmedTracks)
        pos = []; c_map = [];
        for trk = 1:length(confirmedTracks)
            state = confirmedTracks(trk).State;
            p = [state(1), state(3), state(5)]; 
            id = confirmedTracks(trk).TrackID;
            
            pos = [pos; p];
            col_idx = mod(id, 10) + 1;
            c_map = [c_map; colors(col_idx, :)];
            
            allTracksHistory = [allTracksHistory; p, t_curr, id];
            
            fprintf('T=%.2f | ID %d | Pos [%.0f %.0f %.0f]\n', t_curr, id, p);
        end
        
        if ~isempty(pos)
            set(hBirds, 'XData', pos(:,1), 'YData', pos(:,2), 'ZData', pos(:,3), 'CData', c_map);
            drawnow limitrate;
        end
    end
end

%% 5. RÉSULTATS
if ~isempty(allTracksHistory)
    figure('Color','w'); 
    scatter3(micros(:,1), micros(:,2), micros(:,3), 100, 'k^', 'filled'); hold on;
    ids = unique(allTracksHistory(:,5));
    for i=1:length(ids)
        id = ids(i);
        mask = allTracksHistory(:,5)==id;
        pts = allTracksHistory(mask, 1:3);
        plot3(pts(:,1), pts(:,2), pts(:,3), '.-', 'LineWidth', 1.5, 'DisplayName', sprintf('ID %d', id));
    end
    grid on; axis equal; legend; view(3); title('Trajectoires Finales');
    xlim([0 100]); ylim([0 100]); zlim([0 40]);
end


%% --- HELPER FUNCTIONS ---

function filter = initKalmanFilter(detection)
    meas = detection.Measurement;
    
    % Matrice H : On mesure x(1), y(3) et z(5).
    H = [1, 0, 0, 0, 0, 0; 
         0, 0, 1, 0, 0, 0; 
         0, 0, 0, 0, 1, 0];

    filter = trackingKF(...
        'MotionModel', '3D Constant Velocity', ...
        'State', [meas(1); 0; meas(2); 0; meas(3); 0], ...
        'MeasurementModel', H, ... % Matrice numérique
        'StateCovariance', eye(6)*10, ...
        'MeasurementNoise', detection.MeasurementNoise);
end

function tdoa = myHelperMeasureTDOA(state, params)
    micRef = params.MicRefPos;
    micTar = params.MicTarPos;
    pos = state(1:3)';
    c = 343.0;
    dRef = norm(pos - micRef);
    dTar = norm(pos - micTar);
    tdoa = (dTar - dRef) / c;
end

function [state, stateCov] = myHelperTDOA2Pos(detections)
    numDets = numel(detections);
    sensorPosRef = zeros(numDets, 3);
    sensorPosTar = zeros(numDets, 3);
    meas = zeros(numDets, 1);
    weights = zeros(numDets, 1);
    
    for i=1:numDets
        meas(i) = detections{i}.Measurement;
        params = detections{i}.MeasurementParameters;
        sensorPosRef(i,:) = params.MicRefPos;
        sensorPosTar(i,:) = params.MicTarPos;
        weights(i) = 1/detections{i}.MeasurementNoise;
    end
    
    x0 = mean(sensorPosTar, 1);
    c = 343.0;
    
    objective = @(x) sqrt(weights) .* (c * meas - (vecnorm(sensorPosTar - x, 2, 2) - vecnorm(sensorPosRef - x, 2, 2)));
    
    options = optimoptions('lsqnonlin', 'Display', 'off');
    [xSol, ~, residual, ~, ~, ~, jacobian] = lsqnonlin(objective, x0, [], [], options);
    
    state = xSol(:);
    try stateCov = inv(jacobian'*jacobian) * mean(residual.^2); catch stateCov = eye(3); end
end