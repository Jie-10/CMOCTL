function Offspring = Neighbor_Pairing_Strategy2(Problem,MatingPop,Pop,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    Objs = MatingPop.objs;
    [Num,M] = size(Objs);
    Objs = (Objs - repmat(Zmin,Num,1));
    Objs = Objs./repmat(sqrt(sum(Objs.^2,2)),1,M);
    
    Objs1 = Pop.objs;
    [Num2,M] = size(Objs1);
    Objs1 = (Objs1 - repmat(Zmin,Num2,1));
    Objs1 = Objs1./repmat(sqrt(sum(Objs1.^2,2)),1,M);
    
    CosV = Objs * Objs1';
    %     CosV = CosV - 3*eye(Num,Num);
    
    [~,SInd] = sort(-CosV,2);
    
    Nr=min(10,length(Pop));
    %     Nr=min(Num,10);
    Neighbor = SInd(:,1:Nr);
    
    Mate1 = MatingPop;
    
    P = ones(Num,1);
    for i = 1:Num
        P(i) = Neighbor(i,randsample(Nr,1));
    end
    
    Mate2=Pop(P);
    if rand > 0.5
        Offspring=OperatorGAhalf(Problem,[Mate1,Mate2]);
    else
        Offspring=OperatorGAhalf(Problem,[Mate1,Mate2],{1,20,1,1/Problem.D});
    end
end

