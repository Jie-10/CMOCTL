function [MatingPool1, MatingPool2] = TMP(Population, Zmin, flag)
% Two-layer mating pool strategy.


    % Parameters
    Nf = 10;
    Pc = 0.7;

    % Basic data
    PopObj = Population.objs;
    PopCon = Population.cons;
    Num = size(PopObj, 1);

    Nf = min(Nf, Num - 1);
    allIndex = (1:Num)';

    % Objective normalization for distance calculation
    NormObj = PopObj - Zmin;
    NormValue = sqrt(sum(NormObj.^2, 2));
    NormValue(NormValue == 0) = eps;
    NormObj = NormObj ./ NormValue;

    % Distance matrix
    DistMat = pdist2(NormObj, NormObj);
    DistMat(1:Num+1:end) = inf;

    % Non-dominated sorting results
    [FrontNo_c, MaxF_c] = NDSort(NormObj, PopCon, inf);
    [FrontNo_u, MaxF_u] = NDSort(NormObj, inf);

    % Precompute members of each front to avoid repeated find operations
    FrontCell_c = cell(MaxF_c, 1);
    for f = 1 : MaxF_c
        FrontCell_c{f} = find(FrontNo_c == f)';
    end

    FrontCell_u = cell(MaxF_u, 1);
    for f = 1 : MaxF_u
        FrontCell_u{f} = find(FrontNo_u == f)';
    end

    % Parent index matrix
    ParentIndex = zeros(Num, 2);

    % Construct two-layer mating pool for each individual
    for i = 1 : Num
        if flag(i) == 1
            FrontCell = FrontCell_c;
            MaxF = MaxF_c;
        else
            FrontCell = FrontCell_u;
            MaxF = MaxF_u;
        end

        % Construct L1
        L1 = zeros(Nf, 1);
        L1Count = 0;

        for f = 1 : MaxF
            C = FrontCell{f};
            C(C == i) = [];

            if isempty(C)
                continue;
            end

            remain = Nf - L1Count;
            if numel(C) <= remain
                L1(L1Count + 1 : L1Count + numel(C)) = C(:);
                L1Count = L1Count + numel(C);
            else
                [~, rank] = sort(DistMat(i, C), 'ascend');
                chosen = C(rank(1:remain));
                L1(L1Count + 1 : Nf) = chosen(:);
                L1Count = Nf;
                break;
            end

            if L1Count >= Nf
                break;
            end
        end

        L1 = L1(1:L1Count);

        % Construct L2 = P \ (L1 ∪ {x})
        L2Mask = true(Num, 1);
        L2Mask(i) = false;
        L2Mask(L1) = false;
        L2 = allIndex(L2Mask);

        % Robust fallback for very small populations
        if isempty(L1)
            L1 = allIndex(allIndex ~= i);
        end
        if isempty(L2)
            L2 = L1;
        end

        % Select two parents from the same layer
        if rand < Pc
            candidate = L1;
        else
            candidate = L2;
        end

        ParentIndex(i, :) = candidate(randi(numel(candidate), 1, 2));
    end

    % Output mating parents
    MatingPool1 = Population(ParentIndex(:, 1));
    MatingPool2 = Population(ParentIndex(:, 2));
end
