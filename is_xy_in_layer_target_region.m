function yes = is_xy_in_layer_target_region(scene, layerId, xy)
%IS_XY_IN_LAYER_TARGET_REGION Whether xy lies in valid target region.

    [rInner, rOuter] = get_layer_target_region(scene, layerId);
    r = hypot(xy(1), xy(2));

    yes = (r >= rInner) && (r <= rOuter);
end