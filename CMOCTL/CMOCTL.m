classdef CMOCTL < ALGORITHM
% <2026> <multi/many> <real/binary/permutation><constrained/none>

methods
function main(Algorithm, Problem)
    %% Initialize MP and AP
    MP = Problem.Initialization();  
    AP = MP;                        
    FitnessMP = CalFitness(MP.objs, MP.cons, 0);
    FitnessAP = CalFitness(AP.objs);

    %% Parameter initialization
    gen   = 1;
    stage = 1;
    [N, ~] = size(MP.objs);
    alpha = 0.3;
    beta  = 1e-3;
    l     = 50;
    od    = [];
    flagMP = ones(1, N);

    %% Generate weight vectors and define sub-regions
    [W, ~] = UniformPoint(Problem.N, Problem.M);
    CosWW = 1 - pdist2(W, W, 'cosine');
    CosWW = min(max(CosWW, -1), 1);
    AngleW = acos(CosWW);
    AngleW(eye(length(W)) == 1) = inf;
    MinAngle = mean(min(AngleW)) / 2;

    %% Historical average objective values of AP for stage switching
    previousMeanObjAP = mean(AP.objs, 1);

    %% Main loop
    while Algorithm.NotTerminated2(MP,AP)

        %% Parent selection and offspring generation
        Zmin = min([MP.objs; AP.objs], [], 1);

        if stage == 1
            MatingIndexMP = TournamentSelection(2, length(MP), FitnessMP);
            MatingIndexAP = TournamentSelection(2, length(AP), FitnessAP);

            if rand > 0.5
                O1 = OperatorGAhalf(Problem, MP(MatingIndexMP));
                O2 = OperatorGAhalf(Problem, AP(MatingIndexAP));
            else
                O1 = OperatorGAhalf(Problem, MP(MatingIndexMP), {1,20,1,1/Problem.D});
                O2 = OperatorGAhalf(Problem, AP(MatingIndexAP), {1,20,1,1/Problem.D});
            end
        else
                [MatingPoolMP_1, MatingPoolMP_2] = TMP(MP, Zmin, flagMP);
                [MatingPoolAP_1, MatingPoolAP_2] = TMP(AP, Zmin, flagAP);

                O1 = OperatorDE(Problem, MP, MatingPoolMP_1, MatingPoolMP_2);
                O2 = OperatorDE(Problem, AP, MatingPoolAP_1, MatingPoolAP_2);
        end

        %% Stage switching condition
        if mod(gen, l) == 0 && stage == 1
            currentMeanObjAP = mean(AP.objs, 1);
            improvementRate = (previousMeanObjAP - currentMeanObjAP) ./ max(abs(previousMeanObjAP), 1e-6);
            ir = mean(improvementRate);

            if ir <= beta && Problem.FE / Problem.maxFE >= alpha
                meanObjMP = mean(sum(MP.objs, 2));
                meanObjAP = mean(sum(AP.objs, 2));
                od = abs(meanObjMP - meanObjAP) / 2;
                stage = 2;
            else
                previousMeanObjAP = currentMeanObjAP;
            end
        end

        %% Environmental selection
        if stage == 1
            % MP: CDP-based environmental selection
            [MP, FitnessMP] = FeasibilitySelect([MP, O1, O2], N, 0);
            flagMP = ones(1, length(MP));

            % AP: unconstrained environmental selection
            [AP, FitnessAP] = AdvanceSelect([AP, O1, O2], N);
        else
            % MP: CDP-based environmental selection
            [MP, FitnessMP] = FeasibilitySelect([MP, O1, O2], N, 0);

            %% Analyze the distribution of MP using weight vectors
            MPObj = MP.objs;
            CosMPW = 1 - pdist2(MPObj, W, 'cosine');
            CosMPW = min(max(CosMPW, -1), 1);
            AngleMP = acos(CosMPW);
            AngleMP(isnan(AngleMP)) = pi/2;
            InSubspaceMP = AngleMP <= MinAngle;

            % I: sub-regions not occupied by MP
            I = find(sum(InSubspaceMP, 1) == 0);

            % ri: feasible MP individual with the smallest angle to Si
            CV_MP = sum(max(0, MP.cons), 2);
            feasibleMP = find(CV_MP == 0);
            ObjSumMP = sum(MPObj, 2);

            if ~isempty(feasibleMP)
                [~, localIndex] = min(AngleMP(feasibleMP, :), [], 1);
                riIndex = feasibleMP(localIndex);
            else
                % if MP has no feasible solution, use the nearest
                % MP individual as the reference point.
                [~, riIndex] = min(AngleMP, [], 1);
            end
            rObj = ObjSumMP(riIndex);
            rObj = rObj(:)';

            % AP: complementary environmental selection
            Q = [AP, O1, O2];
            [AP, FitnessAP, flagAP] = CES(Q, N, W, MinAngle, I, rObj, od);
        end

        gen = gen + 1;
    end
end
end
end
