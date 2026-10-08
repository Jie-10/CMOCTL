function [Population,Fitness] = FeasibilitySelect(Population,N,epsilon)
% 环境选择：SPEA2-based


    %% 计算适应度
    
    FitnessAll = CalFitness(Population.objs, Population.cons, epsilon);

    %% 环境选择
    Next = FitnessAll < 1;
    if sum(Next) < N
        [~,Rank] = sort(FitnessAll);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs, sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

    % 下一代种群
    Population = Population(Next);
    Fitness    = FitnessAll(Next);

    % 排序
    [Fitness,rank] = sort(Fitness);
    Population     = Population(rank);

end


function Del = Truncation(PopObj,K)
% Truncation：删除K个个体

    Distance = pdist2(real(PopObj),real(PopObj));
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
