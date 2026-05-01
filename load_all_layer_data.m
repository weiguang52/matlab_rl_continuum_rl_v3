function data = load_all_layer_data(dataDir)
%LOAD_ALL_LAYER_DATA Read support/exploration files for four inter-plate spaces.

    names = {'CSP','LGP','SGP','BP'};
    zRanges = [0, 736.42; 736.42, 1714.02; 1714.02, 2209.62; 2209.62, 2783.78];

    data = struct('name', {}, 'zRange', {}, 'supports', {}, 'entries', {});
    for i = 1:numel(names)
        nm = names{i};
        sfile = fullfile(dataDir, sprintf('%s_SupportInfo(7).txt', nm));
        efile = fullfile(dataDir, sprintf('%s_ExplorationInfo(7).txt', nm));
        data(i).name = nm;
        data(i).zRange = zRanges(i,:);
        data(i).supports = parse_support_info(sfile);
        data(i).entries = parse_exploration_info(efile);
    end
end
