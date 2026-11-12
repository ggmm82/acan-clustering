"""
AlgebraicClusterModel & GraphClusterModel (v8).

Provides hierarchical concept graph representation, White-Box concept rule
extraction, and Universal Masked Tabular Modeling (`predict_masked`) capable of
autocompleting arbitrary subsets of missing attributes on any tuple.
"""

import json
import gzip
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_datetime64_any_dtype, is_bool_dtype


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json(orient="index", force_ascii=False))
        elif hasattr(obj, "to_json"):
            return json.loads(obj.to_json())
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)


class AlgebraicClusterModel:
    _ID_COUNTER = 0

    def __init__(self):
        self.id = AlgebraicClusterModel._ID_COUNTER
        AlgebraicClusterModel._ID_COUNTER += 1
        self.n = 0
        self.children = {}
        self.cp = 0.0
        self.Groups = np.empty((0, 0))
        self.Relevance = np.empty((0, 0))
        self.Counts = np.empty((0, 1))
        self.Freqs = np.empty((0, 1))
        self.cols = []

    def k(self):
        return len(self.Groups)

    @staticmethod
    def downsample_classes(df, labels):
        if not labels or labels[0] not in df.columns:
            return df
        lbl = labels[0]
        counts = df[lbl].value_counts()
        min_c = counts.min()
        if min_c < 1:
            return df
        parts = [group.sample(min_c, random_state=42) for _, group in df.groupby(lbl)]
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def summarize(dataset, rootRelevance=None, q=1.0):
        NormDataset = dataset.to_numpy(dtype=np.float64)
        model = AlgebraicClusterModel()
        model.cols = list(dataset.columns.values)
        model.n = len(NormDataset)
        model.RootMin = np.min(NormDataset, axis=0).reshape(1, -1)
        model.RootMax = np.max(NormDataset, axis=0).reshape(1, -1)
        model.Root = np.mean(NormDataset, axis=0).reshape(1, -1)
        model.m = len(model.cols)
        model.RootError = np.std(NormDataset, axis=0).reshape(1, -1)
        model.q = q
        if rootRelevance is None:
            model.RootRelevance = np.ones_like(model.Root)
        else:
            model.RootRelevance = np.asarray(rootRelevance).reshape(1, -1)
        return model

    def fullLabels(self, labels):
        fullLabels = []
        for col in self.cols:
            for lbs in labels:
                if lbs == col or col.startswith(str(lbs) + "_"):
                    if col not in fullLabels:
                        fullLabels.append(col)
        return fullLabels

    def getD(self, t, cix):
        G = self.Groups[:, cix]
        return G - t

    def getE(self, weights, t, cix):
        E = weights * self.getD(t, cix)
        E2 = np.sum(E * E, axis=1, keepdims=True)
        return np.sqrt(E2)

    def predictTuple(self, rowTuple, cix):
        """
        v7-compatible hard cluster assignment using linear row-normalized Relevance R.
        """
        if getattr(self,'routing_profile','legacy-v8') in ('paper-routing-v1','cp-routing-v1','cp-joint-v1','cp-v9'):
            weights, confidence, best = self.predictTupleSoft(rowTuple,cix)
            return best, float(-np.log(max(confidence,1e-12)))
        t = np.asarray(rowTuple, dtype=np.float64).reshape(1, -1)
        w = self.Relevance[:, cix].copy()
        row_sums = w.sum(axis=1, keepdims=True)
        w = np.divide(w, row_sums, out=np.ones_like(w) / max(len(cix), 1), where=row_sums > 1e-12)
        E = self.getE(w, t, cix)
        k_best = int(np.argmin(E[:, 0]))
        return k_best, float(E[k_best, 0])

    def predictTupleSoft(self, rowTuple, cix, gamma=1.5):
        """
        ACAN v8 Sharpened Feature-Relevance Attention + Local Weighted Concept Inference.
        Raises feature relevance R to power `gamma` and applies a soft threshold against
        spurious finite-sample noise on irrelevant dimensions (Theorem 4.1.1 continuous analog).
        """
        if getattr(self,'routing_profile','legacy-v8') == 'cp-routing-v1':
            from .variational import predict
            return predict(self,rowTuple,cix)
        if getattr(self,'routing_profile','legacy-v8') == 'cp-v9':
            from .concept_adapter import predict
            return predict(self,rowTuple,cix)
        if getattr(self,'routing_profile','legacy-v8') == 'cp-joint-v1':
            from .supervised import predict
            return predict(self,rowTuple,cix)
        if getattr(self,'routing_profile','legacy-v8') == 'paper-routing-v1':
            from acan.algebra import routing_logits, mass_modulation
            t = np.asarray(rowTuple,dtype=np.float64).reshape(1,-1)
            p = self.routing_parameters
            logits = routing_logits(t,self.Groups[:,cix],self.Relevance[:,cix],p['alpha'],p['beta'])
            frequencies = np.asarray(self.Freqs).reshape(-1)
            logits = mass_modulation(logits,frequencies,p['gamma'],np.log)
            weights = np.exp(logits-logits.max());weights /= weights.sum()
            best = int(weights.argmax())
            return weights, float(weights[best]), best
        t = np.asarray(rowTuple, dtype=np.float64).reshape(1, -1)
        w_raw = self.Relevance[:, cix].copy()
        # Suppress low-relevance noise floor (< 35% of each slot's peak relevance)
        row_max = np.max(w_raw, axis=1, keepdims=True)
        w_clean = np.where(w_raw >= 0.15 * row_max, w_raw, 0.0)
        w_sharp = np.power(w_clean, gamma)
        row_sums = w_sharp.sum(axis=1, keepdims=True)
        w = np.divide(w_sharp, row_sums, out=np.ones_like(w_sharp) / max(len(cix), 1), where=row_sums > 1e-12)
        E = self.getE(w, t, cix)[:, 0]

        inv_d2 = 1.0 / (np.square(E) + 1e-5)
        freqs = self.Freqs.reshape(-1) if self.Freqs is not None and len(self.Freqs) == len(E) else np.ones_like(E)
        scores = inv_d2 * np.power(np.clip(freqs, 1e-4, None), 0.35)
        if len(scores) > 3:
            top_idx = np.argsort(scores)[-3:]
            mask = np.zeros_like(scores)
            mask[top_idx] = 1.0
            scores = scores * mask
        weights = scores / np.clip(np.sum(scores), 1e-12, None)
        min_err = float(np.min(E))
        conf = float(np.exp(-min_err))
        return weights, conf, int(np.argmax(weights))

    def assignCluster(self, tuples, cix):
        if len(tuples) == 0:
            return np.empty((0, 2))
        results = [self.predictTuple(row, cix) for row in tuples]
        return np.array(results, dtype=np.float64)

    def splitDataset(self, normDataset):
        cols = self.cols
        intCols = [c for c in cols if c in normDataset.columns]
        cix = [cols.index(col) for col in intCols]
        k = len(self.Groups)
        tuples = normDataset[intCols].to_numpy(dtype=np.float64)
        res = {}
        preds = self.assignCluster(tuples, cix)
        for i in range(k):
            ix = np.argwhere(preds[:, 0] == i)
            npD = normDataset.values[ix[:, 0], :]
            currDataset = pd.DataFrame(npD, columns=normDataset.columns, index=normDataset.index[ix[:, 0]])
            res[i] = (currDataset, ix)
        return res

    def predictValue(self, flbs):
        R = pd.DataFrame(self.Root, columns=self.cols)
        RE = pd.DataFrame(self.RootError, columns=self.cols)
        pos = R[flbs].idxmax(axis=1).iloc[0]
        val = float(R.loc[0, pos])
        err = float(2.0 * RE.loc[0, pos])
        return val, err, pos

    def extractRules(self, minMax=None, top_features=4):
        rules = []
        for k_idx in range(len(self.Groups)):
            g_row = self.Groups[k_idx]
            r_row = self.Relevance[k_idx]
            count_k = int(self.Counts[k_idx, 0]) if len(self.Counts) > k_idx else 0
            freq_k = float(self.Freqs[k_idx, 0]) if len(self.Freqs) > k_idx else 0.0
            top_idx = np.argsort(r_row)[::-1][:top_features]
            conditions = []
            for c_i in top_idx:
                col_name = self.cols[c_i]
                val_norm = g_row[c_i]
                if minMax and col_name in minMax:
                    vmin, vmax = minMax[col_name]
                    val_real = val_norm * (vmax - vmin) + vmin
                else:
                    val_real = val_norm
                conditions.append({
                    "attribute": col_name,
                    "centroid_value": round(float(val_real), 4),
                    "normalized_value": round(float(val_norm), 4),
                    "relevance": round(float(r_row[c_i]), 4),
                })
            rules.append({
                "concept_id": f"Node{self.id}_G{k_idx}",
                "support_count": count_k,
                "support_freq": round(freq_k, 4),
                "top_attributes": conditions,
            })
        return rules

    @staticmethod
    def euclideanDistance(pA, pB):
        d = pA - pB
        return np.sqrt(np.sum(d * d, axis=1, keepdims=True))

    @staticmethod
    def loadNode(dictsModel):
        model = AlgebraicClusterModel()
        model.cols = dictsModel["cols"]
        model.id = dictsModel["id"]
        model.Root = np.asarray(dictsModel["Root"], dtype=np.float64)
        model.RootError = np.asarray(dictsModel["RootError"], dtype=np.float64)
        model.n = dictsModel["n"]
        model.Relevance = np.asarray(dictsModel["Relevance"], dtype=np.float64)
        model.Groups = np.asarray(dictsModel["Groups"], dtype=np.float64)
        model.cp = dictsModel["cp"]
        model.routing_profile = dictsModel.get('routing_profile','legacy-v8')
        model.routing_parameters = dictsModel.get('routing_parameters')
        model.Freqs = np.asarray(dictsModel["Freqs"], dtype=np.float64)
        model.Counts = np.asarray(dictsModel["Counts"], dtype=np.float64)
        model.children = {int(k): v for k, v in dictsModel.get("children", {}).items()}
        return model

    def to_json(self):
        return json.dumps(self.__dict__, cls=ComplexEncoder, indent=4)


class GraphClusterModel:
    def __init__(self):
        self.clusterModel = {}
        self.clusterData = {}
        self.labels = []
        self.rootId = 0
        self.minMax = {}
        self.preprocessing = None

    def to_json(self):
        cd = self.clusterData
        self.clusterData = None
        self.clusterCount = len(self.clusterModel)
        js = json.dumps(self.__dict__, cls=ComplexEncoder, indent=4)
        self.clusterData = cd
        return js

    @staticmethod
    def loadModel(strModel=None):
        gcm = GraphClusterModel()
        dictsModel = json.loads(
            strModel,
            object_hook=lambda d: {int(k) if str(k).isdigit() else k: v for k, v in d.items()},
        )
        gcm.labels = dictsModel["labels"]
        gcm.preprocessing = dictsModel.get("preprocessing")
        gcm.rootId = dictsModel["rootId"]
        gcm.minMax = {str(k): v for k, v in dictsModel["minMax"].items()}
        for ids, model in dictsModel["clusterModel"].items():
            gcm.clusterModel[int(ids)] = AlgebraicClusterModel.loadNode(model)
        return gcm

    def evaluateError(self, dataset, labels, use_soft_attention=True):
        root = self.clusterModel[self.rootId]
        n = len(dataset)
        fullLabels = root.fullLabels(labels)
        X_test = dataset.drop(labels, axis=1, errors="ignore")
        Y_test_full = GraphClusterModel.full_dummies(dataset[labels])
        for col in fullLabels:
            if col not in Y_test_full.columns:
                Y_test_full[col] = 0.0
        Y_test = Y_test_full[fullLabels]
        Y_pred, Y_conf, path = self.predictDataset(X_test, labels, use_soft_attention=use_soft_attention)

        errors = AlgebraicClusterModel.euclideanDistance(Y_test.to_numpy(dtype=np.float64), Y_pred.to_numpy(dtype=np.float64))
        error = float(np.sum(errors))
        m = max(len(fullLabels), 1)
        den = n * (m ** 0.5)
        return error / den, Y_test, Y_pred, Y_conf, path

    def predictDataset(self, dataset, labels, use_soft_attention=True):
        root = self.clusterModel[self.rootId]
        fullLabels = root.fullLabels(labels)
        cols = [c for c in self.minMax.keys() if c not in fullLabels]
        fullDataset = GraphClusterModel.full_dummies(dataset)
        normDataset = self.alignData(fullDataset, cols)
        res, conf, path = self.assignLabel(root, normDataset, fullLabels, use_soft_attention=use_soft_attention)
        Y_pred = pd.DataFrame(res, columns=fullLabels, index=normDataset.index)
        Y_conf = pd.DataFrame(conf, columns=fullLabels, index=normDataset.index)
        return Y_pred, Y_conf, path

    def assignLabel(self, node, normDataset, fullLabels, use_soft_attention=True):
        if (use_soft_attention and getattr(node,'routing_profile',None) in ('cp-joint-v1','cp-v9')
                and (node.routing_parameters or {}).get('hierarchy_readout')=='conditional-mixture-v1'):
            return self._assign_joint_mixture(node,normDataset,fullLabels)
        res = np.zeros((len(normDataset), len(fullLabels)), dtype=np.float64)
        conf = np.zeros((len(normDataset), len(fullLabels)), dtype=np.float64)
        path = np.empty(len(normDataset), dtype="object")

        G_df = pd.DataFrame(node.Groups, columns=node.cols)
        target_G = G_df[fullLabels].to_numpy(dtype=np.float64)

        if use_soft_attention and len(node.children) == 0:
            intCols = [c for c in node.cols if c in normDataset.columns and c not in fullLabels]
            cix = [node.cols.index(c) for c in intCols]
            tuples = normDataset[intCols].to_numpy(dtype=np.float64)
            for row_i in range(len(tuples)):
                w_k, c_val, best_k = node.predictTupleSoft(tuples[row_i], cix)
                res[row_i] = np.matmul(w_k, target_G)
                conf[row_i] = c_val
                path[row_i] = f"{node.id}_G{best_k}"
            return res, conf, path

        k = len(node.Groups)
        splits = node.splitDataset(normDataset)
        intCols = [c for c in node.cols if c in normDataset.columns and c not in fullLabels]
        cix = [node.cols.index(c) for c in intCols]

        for i in range(k):
            ix = splits[i][1]
            if len(ix) == 0:
                continue
            if i not in node.children:
                res[ix[:, 0]] = target_G[i]
                sub_tuples = splits[i][0][intCols].to_numpy(dtype=np.float64)
                for r_idx, global_idx in enumerate(ix[:, 0]):
                    _, dist_err = node.predictTuple(sub_tuples[r_idx], cix)
                    conf[global_idx] = np.exp(-dist_err)
                path[ix[:, 0]] = f"{node.id}_G{i}"
            else:
                currDataset = splits[i][0]
                if len(currDataset) > 0:
                    childNode = self.clusterModel[node.children[i]]
                    res[ix[:, 0]], conf[ix[:, 0]], path[ix[:, 0]] = self.assignLabel(
                        childNode, currDataset, fullLabels, use_soft_attention=use_soft_attention
                    )
        return res, conf, path

    def _assign_joint_mixture(self, node, dataset, labels):
        """Marginalize conditional concept paths instead of choosing one branch.

        A child's prediction replaces its parent's readout, weighted by the
        parent's posterior. The reported path is the most probable complete
        latent path; its probability is not a calibrated class confidence.
        Bounded batches keep the routing workspace independent of query size.
        """
        if getattr(node,'routing_profile',None)=='cp-v9':
            from .concept_adapter import predict_assignments
        else:
            from .supervised import predict_assignments
        if len(dataset)>256:
            chunks=[self._assign_joint_mixture(node,dataset.iloc[start:start+256],labels)
                    for start in range(0,len(dataset),256)]
            return tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(3))
        columns=[c for c in node.cols if c in dataset.columns and c not in labels]
        cix=[node.cols.index(c) for c in columns]
        weights=predict_assignments(node,dataset[columns].to_numpy(dtype=np.float64),cix)
        readout=pd.DataFrame(node.Groups,columns=node.cols)[labels].to_numpy(dtype=np.float64)
        terminal=weights.copy()
        if node.children:
            terminal[list(node.children)]=0
        result=terminal.T@readout
        best=terminal.argmax(axis=0)
        confidence=terminal.max(axis=0)
        paths=np.asarray([f'{node.id}_G{i}' for i in best],dtype=object)
        for slot,child_id in node.children.items():
            child_result,child_confidence,child_paths=self._assign_joint_mixture(
                self.clusterModel[child_id],dataset,labels)
            result+=weights[slot,:,None]*child_result
            path_probability=weights[slot]*child_confidence[:,0]
            replace=path_probability>confidence
            confidence[replace]=path_probability[replace]
            paths[replace]=child_paths[replace]
        return result,np.repeat(confidence[:,None],len(labels),axis=1),paths

    def predict_masked(self, incomplete_df):
        root = self.clusterModel[self.rootId]
        completed_df = incomplete_df.copy()

        # A CSV column containing only blanks is normally inferred as float64.
        # Convert according to the trained schema before inserting predictions.
        for col in completed_df.columns:
            targets = root.fullLabels([col])
            if completed_df[col].isna().any() and targets:
                if targets != [col]:
                    completed_df[col] = completed_df[col].astype(object)
                else:
                    completed_df[col] = completed_df[col].astype(float)

        for idx, row in incomplete_df.iterrows():
            missing_cols = [c for c in incomplete_df.columns if pd.isna(row[c])]
            if not missing_cols:
                continue
            known_cols = [c for c in incomplete_df.columns if not pd.isna(row[c])]
            known_df = pd.DataFrame([row[known_cols].to_dict()])
            known_dummies = GraphClusterModel.full_dummies(known_df)
            norm_known = self.alignData(known_dummies, list(known_dummies.columns))

            valid_known = [c for c in norm_known.columns if c in root.cols]
            if not valid_known:
                continue
            cix = [root.cols.index(c) for c in valid_known]
            row_vec = norm_known[valid_known].to_numpy(dtype=np.float64)[0]

            w_k, _, _ = root.predictTupleSoft(row_vec, cix)
            inferred_full = np.matmul(w_k, root.Groups)
            inferred_series = pd.Series(inferred_full, index=root.cols)

            for mcol in missing_cols:
                m_dummies = root.fullLabels([mcol])
                if len(m_dummies) == 1 and m_dummies[0] == mcol:
                    vmin, vmax = self.minMax.get(mcol, (0.0, 1.0))
                    completed_df.loc[idx, mcol] = inferred_series[mcol] * (vmax - vmin) + vmin
                elif m_dummies:
                    best_cat_col = inferred_series[m_dummies].idxmax()
                    cat_val = best_cat_col[len(str(mcol)) + 1 :]
                    completed_df.loc[idx, mcol] = cat_val
        return completed_df

    def predictValues(self, Y_Pred, Y_Conf, path):
        df = pd.DataFrame()
        Y_Pred = self.denormalizeData(Y_Pred)
        pos = Y_Pred.values.argmax(axis=1)
        ind = np.arange(len(pos))
        val = Y_Pred.values[ind, pos]
        conf = Y_Conf.values[ind, pos]
        df["Pred_Pos"] = Y_Pred.columns[pos]
        df["Pred_Val"] = val
        df["Pred_Conf"] = conf
        df["Pred_Path"] = path
        return df

    def alignData(self, fullDataset, cols):
        norm_dict = {}
        for col in cols:
            if col in fullDataset.columns.values and col in self.minMax:
                max_value = self.minMax[col][1]
                min_value = self.minMax[col][0]
                if max_value == min_value:
                    norm_dict[col] = np.zeros(len(fullDataset), dtype=np.float64)
                else:
                    norm_dict[col] = ((fullDataset[col] - min_value) / (max_value - min_value)).to_numpy(dtype=np.float64)
        return pd.DataFrame(norm_dict, index=fullDataset.index)

    @staticmethod
    def full_dummies(dataset):
        return pd.get_dummies(GraphClusterModel.full_cols(dataset), drop_first=False, dtype=float).copy()

    @staticmethod
    def full_cols(dataset):
        assert isinstance(dataset, pd.DataFrame)
        dataset = dataset.copy().dropna(axis=1, how="all")
        for col in dataset.columns:
            if dataset[col].isnull().sum() > 0:
                if is_string_dtype(dataset[col]) or dataset[col].dtype == object:
                    dataset[col] = dataset[col].replace(np.nan, "_NaN")
                else:
                    newCol = str(col) + "_NaN"
                    dataset[newCol] = np.where(dataset[col].isnull(), 1.0, 0.0)
                    dataset[col] = dataset[col].fillna(dataset[col].mean())

            if is_bool_dtype(dataset[col]):
                dataset[col] = dataset[col] * 1.0
            elif is_datetime64_any_dtype(dataset[col]):
                dataset[col + "_day"] = dataset[col].map(lambda x: x.day)
                dataset[col + "_month"] = dataset[col].map(lambda x: x.month)
                dataset[col + "_year"] = dataset[col].map(lambda x: x.year)
                del dataset[col]
        return dataset

    @staticmethod
    def normalizeData(dataset):
        cols = list(dataset.columns.values)
        minMax = {}
        norm_dict = {}
        for col in cols:
            max_value = float(dataset[col].max())
            min_value = float(dataset[col].min())
            minMax[col] = (min_value, max_value)
            if max_value == min_value:
                norm_dict[col] = np.zeros(len(dataset), dtype=np.float64)
            else:
                norm_dict[col] = ((dataset[col] - min_value) / (max_value - min_value)).to_numpy(dtype=np.float64)
        return pd.DataFrame(norm_dict, index=dataset.index), minMax

    def denormalizeData(self, dataset):
        result = dataset.copy()
        for col in dataset.columns.values:
            if col in self.minMax:
                max_value = self.minMax[col][1]
                min_value = self.minMax[col][0]
                result[col] = (dataset[col] * (max_value - min_value)) + min_value
        return result
