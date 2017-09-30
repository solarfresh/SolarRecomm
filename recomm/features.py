def index_single_feature(feature_list):
    index_count = -1
    collect_feature = []
    index_feature = []
    for item in feature_list:
        if item in collect_feature:
            index_feature.append(collect_feature.index(item))
        else:
            index_count += 1
            index_feature.append(index_count)
            collect_feature.append(item)

    return index_feature, collect_feature
