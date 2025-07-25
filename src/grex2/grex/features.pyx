import itertools
import collections
import numpy as np
import scipy.sparse

from .utils import Dict


# Should not be used,
# I implemented this because unregularized intercept term
# was not implemented in celer. However, as I now use skglm,
# this is useless...
class InterceptFeature:
    def __init__(self):
        self.initialized = True

    def init_from_data(self, data):
        pass

    def build_features(self, X, data, offset):
        X[:, offset] = 1

    def get_all_names(self):
        return ["intercept"]

    def __len__(self):
        return 1


class ClassFeature:
    def __init__(self, name):
        self.name = name
        self.initialized = False

    def init_from_data(self, data):
        values = set()
        for sentence in data:
            if self.name in sentence:
                v = sentence[self.name]
                assert type(v) == str
                values.add(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, X, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                value = dep[self.name]
                value_id = self.dict.str_to_id(value)
                X[i, offset + value_id] = 1

    def get_all_names(self):
        return ["%s=%s" % (self.name, v) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            return len(self.dict)


class IndicatorFeature:
    def __init__(self, name):
        self.name = name
        self.initialized = False

    def init_from_data(self, data):
        values = set()
        for dep in data:
            if self.name in dep:
                v = dep[self.name]
                assert type(v) == set
                values.update(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, X, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                for value in dep[self.name]:
                    value_id = self.dict.str_to_id(value)
                    X[i, offset + value_id] = 1

    def get_all_names(self):
        return ["%s=%s" % (self.name, v) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return len(self.dict)


class AllSingletonFeatures:
    def __init__(self, predicate=None):
        self.predicate = predicate
        self.initialized = False

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        for sentence in data:
            for k, v in sentence.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if type(v) == str:
                    class_feature_names.add(k)
                elif type(v) == set:
                    indicator_feature_names.add(k)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if len(class_feature_names.intersection(indicator_feature_names)) != 0:
            raise RuntimeError("Error in feature types")

        self.features = list()
        self.len_ = 0
        for name in class_feature_names:
            feature = ClassFeature(name)
            feature.init_from_data(data)
            self.len_ += len(feature)
            self.features.append(feature)

        for name in indicator_feature_names:
            feature = IndicatorFeature(name)
            feature.init_from_data(data)
            self.len_ += len(feature)
            self.features.append(feature)
        self.initialized = True

    def build_features(self, X, data, offset):
        offset2 = 0
        for feature in self.features:
            feature.build_features(X, data, offset + offset2)
            offset2 += len(feature)

    def get_all_names(self):
        return itertools.chain(*[feature.get_all_names() for feature in self.features])

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.len_


class AllProductFeatures:
    def __init__(self, degree=2, weight=1, min_occurences=1, predicate=None):
        self.predicate = predicate
        self.initialized = False
        self.degree = degree
        self.min_occurences = min_occurences
        self.weight = weight

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        for dep in data:
            for k, v in dep.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if type(v) == str:
                    class_feature_names.add(k)
                elif type(v) == set:
                    indicator_feature_names.add(k)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if len(class_feature_names.intersection(indicator_feature_names)) != 0:
            raise RuntimeError("Error in feature types")

        merged_features = {(False, k) for k in class_feature_names}.union((True, k) for k in indicator_feature_names)
        feature_count = collections.defaultdict(lambda: 0)
        for dep in data:
            for names in itertools.combinations(merged_features, self.degree):
                # This is new: I skip the interaction between child's features
                #if sum('child' in name for _, name in names) > 1:
                #    continue
                all_keys = []
                for is_set_feature, name in names:
                    if name not in dep:
                        break  # cannot build feature for this dep
                    if is_set_feature:
                        if len(all_keys) == 0:
                            all_keys = [[(name, v)] for v in dep[name]]
                        else:
                            all_keys = [
                                k + [(name, v)]
                                for k in all_keys
                                for v in dep[name]
                            ]
                    else:
                        if len(all_keys) == 0:
                            all_keys = [[(name, dep[name])]]
                        else:
                            all_keys = [k + [(name, dep[name])] for k in all_keys]
                else:
                    # exit without break
                    for key in all_keys:
                        key = tuple(key)
                        feature_count[key] += 1

        # filter on number of occurences
        if self.min_occurences > 1:
            filtered_features = list(k for k, v in feature_count.items() if v >= self.min_occurences)
        else:
            filtered_features = list(feature_count.keys())

        self.templates = dict()
        self.n_features = 0
        for feature in filtered_features:
            ks = tuple(k for k, _ in feature)
            vs = tuple(v for _, v in feature)
            if ks not in self.templates:
                self.templates[ks] = {vs: self.n_features}
            elif vs not in self.templates[ks]:  # is this test really necessary?!
                self.templates[ks][vs] = self.n_features
            else:
                print(feature)
                print(ks)
                print(self.templates[ks])
                raise RuntimeError("SHOULD NOT HAPPEN")
            self.n_features += 1

        self.initialized = True
        self.valid_features = filtered_features

    def build_features(self, X, data, int offset):
        cdef int i, j

        for i, dep in enumerate(data):
            for ks, ks_vs in self.templates.items():
                instance_values = list()
                valid = True
                for k in ks:
                    if k not in dep:
                        valid = False
                        break
                    elif type(dep[k]) == str:
                        if len(instance_values) == 0:
                            instance_values = [[dep[k]]]
                        else:
                            instance_values = [vs + [dep[k]] for vs in instance_values]
                    else:
                        if len(instance_values) == 0:
                            instance_values = [[v] for v in dep[k]]
                        else:
                            instance_values = [vs + [v] for vs in instance_values for v in dep[k]]

                if valid:
                    for vs in instance_values:
                        j = ks_vs.get(tuple(vs), -1)
                        if j >= 0:
                            X[i, offset + j] = self.weight

    def get_all_names(self):
        return [",".join("%s=%s" % (k, v) for k, v in feature) for feature in self.valid_features]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.n_features


class FeatureSet:
    def __init__(self):
        self.features = list()

    def add_feature(self, feature):
        self.features.append(feature)

    def init_from_data(self, data):
        for feature in self.features:
            feature.init_from_data(data)

    def build_features(self, data, sparse=True):
        n_columns = sum(len(f) for f in self.features)
        if sparse:
            # TODO: check in celer what kind of sparse matrix it can use
            X = scipy.sparse.lil_matrix((len(data), n_columns))
        else:
            X = np.zeros((len(data), n_columns))

        offset = 0
        for feature in self.features:
            feature.build_features(X, data, offset)
            offset += len(feature)

        if sparse:
            X = scipy.sparse.csc_matrix(X)
        return X

    def feature_weights(self, weights, keep_cols=None, ignore_zeros=True):
        if keep_cols is None:
            keep_cols = []
        ret = dict()
        offset = 0
        for feature in self.features:
            names = feature.get_all_names()
            for i, name in enumerate(names):
                if not ignore_zeros or not np.isclose(weights[offset], 0):
                    ret[name] = (weights[offset], offset)
                offset += 1
        return ret

    def print_weights(self, weights, ignore_zeros=True):
        for n, (v, _) in self.feature_weights(weights, ignore_zeros=ignore_zeros).items():
            print("%s:\t%.4f" % (n, v))
