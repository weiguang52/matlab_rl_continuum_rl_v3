classdef ReplayBuffer < handle
%REPLAYBUFFER 经验回放缓冲区（循环队列）
%
%  用法:
%    buf = ReplayBuffer(capacity, stateDim)
%    buf.push(s, a, r, s2, done)
%    [s, a, r, s2, done] = buf.sample(batchSize)

    properties
        cap       % 容量
        sDim      % 状态维度
        ptr       % 当前写指针（1-based）
        size_     % 当前有效条目数

        S         % stateDim x cap
        A         % 1 x cap  (int32)
        R         % 1 x cap
        S2        % stateDim x cap
        D         % 1 x cap  (done flag, 0/1)
    end

    methods
        function obj = ReplayBuffer(capacity, stateDim)
            obj.cap  = capacity;
            obj.sDim = stateDim;
            obj.ptr  = 1;
            obj.size_ = 0;

            obj.S  = zeros(stateDim, capacity, 'single');
            obj.A  = zeros(1, capacity, 'int32');
            obj.R  = zeros(1, capacity, 'single');
            obj.S2 = zeros(stateDim, capacity, 'single');
            obj.D  = zeros(1, capacity, 'single');
        end

        function push(obj, s, a, r, s2, done)
            p = obj.ptr;
            obj.S(:, p)  = single(s(:));
            obj.A(p)     = int32(a);
            obj.R(p)     = single(r);
            obj.S2(:, p) = single(s2(:));
            obj.D(p)     = single(done);

            obj.ptr  = mod(p, obj.cap) + 1;
            obj.size_ = min(obj.size_ + 1, obj.cap);
        end

        function [s, a, r, s2, done] = sample(obj, batch)
            idx = randperm(obj.size_, min(batch, obj.size_));
            s    = double(obj.S(:, idx));
            a    = double(obj.A(idx));
            r    = double(obj.R(idx));
            s2   = double(obj.S2(:, idx));
            done = double(obj.D(idx));
        end

        function n = size(obj)
            n = obj.size_;
        end
    end
end
