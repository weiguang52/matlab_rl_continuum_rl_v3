classdef SACReplayBuffer < handle
%SACREPLAYBUFFER 经验回放缓冲区（支持连续动作）

    properties
        cap
        sDim
        aDim
        ptr
        size_
        S
        A
        R
        S2
        D
    end

    methods
        function obj = SACReplayBuffer(capacity, stateDim, actDim)
            obj.cap   = capacity;
            obj.sDim  = stateDim;
            obj.aDim  = actDim;
            obj.ptr   = 1;
            obj.size_ = 0;

            obj.S  = zeros(stateDim, capacity, 'single');
            obj.A  = zeros(actDim,   capacity, 'single');
            obj.R  = zeros(1,        capacity, 'single');
            obj.S2 = zeros(stateDim, capacity, 'single');
            obj.D  = zeros(1,        capacity, 'single');
        end

        function push(obj, s, a, r, s2, done)
            p = obj.ptr;
            obj.S(:,p)  = single(s(:));
            obj.A(:,p)  = single(a(:));
            obj.R(p)    = single(r);
            obj.S2(:,p) = single(s2(:));
            obj.D(p)    = single(done);

            obj.ptr   = mod(p, obj.cap) + 1;
            obj.size_ = min(obj.size_ + 1, obj.cap);
        end

        function [s, a, r, s2, done] = sample(obj, batch)
            idx  = randperm(obj.size_, min(batch, obj.size_));
            s    = double(obj.S(:,idx));
            a    = double(obj.A(:,idx));
            r    = double(obj.R(idx));
            s2   = double(obj.S2(:,idx));
            done = double(obj.D(idx));
        end

        function n = size(obj)
            n = obj.size_;
        end
    end
end
