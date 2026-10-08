function [AP, FitnessAP, flagAP] = CES(Q, N, W, MinAngle, I, rObj, od)
% Complementary environmental selection strategy.

    % Basic data
    NumW = size(W, 1);
    NumQ = length(Q);
    QObj = Q.objs;
    QCon = Q.cons;
    QObjSum = sum(QObj, 2);

    % Assign Q to sub-regions according to the angle to weight vectors
    CosQW = 1 - pdist2(QObj, W, 'cosine');
    CosQW = min(max(CosQW, -1), 1);
    AngleQ = acos(CosQW);
    AngleQ(isnan(AngleQ)) = pi/2;
    InSubspaceQ = AngleQ <= MinAngle;

    % Calculate constrained and unconstrained fitness values of Q
    fitc = CalFitness(QObj, QCon, 0); 
    fitu = CalFitness(QObj);           

    % I records the sub-regions not occupied by MP
    if isempty(I)
        I = 1 : NumW;
    end
    I = I(:)';

    % Preallocate selected results to avoid repeatedly concatenating objects
    selectedIndex   = zeros(1, N);
    selectedFitness = zeros(1, N);
    flagAP          = ones(1, N);
    selected        = false(1, NumQ);
    selectedCount   = 0;

    % Iteratively select individuals for AP
    while selectedCount < N
        for i = I
            if selectedCount >= N
                break;
            end

            % R
            R = find(~selected);
            if isempty(R)
                break;
            end

            % Ti
            Ti = find(InSubspaceQ(:, i)' & ~selected);

            if isempty(Ti)
                % Rule 1
                [~, bestLocal] = min(AngleQ(R, i));
                x = R(bestLocal);
                currentFitness = fitc(x);
                currentFlag = 1;
            else
                meanObjTi = mean(QObjSum(Ti));
                meanObjRi = rObj(i);

                if isnan(meanObjRi)
                    meanObjRi = meanObjTi;
                end

                delta_i = abs(meanObjTi - meanObjRi);

                % Rules 2-4 in the CES strategy
                if delta_i <= od
                    % Rule 2
                    candidateFitness = fitc;
                    currentFlag = 1;
                else
                    if meanObjTi > meanObjRi
                        % Rule 3
                        candidateFitness = fitu;
                        currentFlag = 2;
                    else
                        % Rule 4
                        candidateFitness = fitc;
                        currentFlag = 1;
                    end
                end

                [~, bestLocal] = min(candidateFitness(Ti));
                x = Ti(bestLocal);
                currentFitness = candidateFitness(x);
            end

            % Update AP_new
            selectedCount = selectedCount + 1;
            selectedIndex(selectedCount)   = x;
            selectedFitness(selectedCount) = currentFitness;
            flagAP(selectedCount)          = currentFlag;
            selected(x) = true;
        end
    end

    % Output selected AP
    selectedIndex = selectedIndex(1:selectedCount);
    AP = Q(selectedIndex);
    FitnessAP = selectedFitness(1:selectedCount);
    flagAP = flagAP(1:selectedCount);
end
