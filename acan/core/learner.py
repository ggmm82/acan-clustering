"""
AlgebraicClusterLearner & ACANPredictor (v8).

Implements gradient-based optimization of Cluster Probability (cp) with Adam,
automatic concept pruning (k_active << n), hierarchical recursive expansion
(gLearn), and Early Stopping with patience.
"""

import threading
import time
import numpy as np
import pandas as pd
from .network import ModularClusterNet, _AnalyticalClusterNet, HAS_TORCH
from .model import AlgebraicClusterModel, GraphClusterModel

if HAS_TORCH:
    import torch
    import torch.optim as optim


class _NumpyAdam:
    """Standard Adam optimizer (Kingma & Ba, 2015) for AnalyticalClusterNet parameters."""

    def __init__(self, params_dict, lr=0.08, betas=(0.9, 0.999), eps=1e-8):
        self.params = params_dict
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params_dict.items()}
        self.v = {k: np.zeros_like(v) for k, v in params_dict.items()}
        self.t = 0
        self._root_second_moments = set()

    def step(self, grads_dict):
        self.t += 1
        for k, p in self.params.items():
            g = grads_dict[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            # Keep the ordinary path bit-identical until squaring a finite
            # gradient would overflow. Then represent sqrt(v) with hypot;
            # this is the same Adam update, without clipping its direction.
            if k not in self._root_second_moments and np.max(np.abs(g),initial=0)>np.sqrt(np.finfo(g.dtype).max):
                self.v[k] = np.sqrt(self.v[k])
                self._root_second_moments.add(k)
            if k in self._root_second_moments:
                self.v[k] = np.hypot(np.sqrt(self.b2)*self.v[k],np.sqrt(1-self.b2)*g)
            else:
                self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[k] / (1.0 - self.b1 ** self.t)
            v_root = (self.v[k]/np.sqrt(1-self.b2**self.t) if k in self._root_second_moments
                      else np.sqrt(self.v[k]/(1-self.b2**self.t)))
            p -= self.lr * m_hat / (v_root + self.eps)


class AlgebraicClusterLearner:
    """
    Scalable Learner for Algebraic Concept Attention Network (v8).
    """

    def __init__(
        self,
        ds,
        q=1.0,
        batch_size=300,
        device="cpu",
        lr=0.08,
        tau=0.20,
        alpha=0.5,
        labels=None,
        max_k=None,
        random_state=42,
        routing_profile="legacy-v8",
    ):
        if routing_profile not in ('legacy-v8','paper-routing-v1','cp-routing-v1','cp-joint-v1','cp-v9','concept-graph-v1'):
            raise ValueError('Profilo matematico non riconosciuto.')
        self.routing_profile = routing_profile
        self.full_dataset = ds.copy()
        self.total_n = len(self.full_dataset)
        actual_batch_size = min(batch_size, self.total_n)
        if routing_profile in ("cp-routing-v1", "cp-joint-v1", "cp-v9", "concept-graph-v1"):
            actual_batch_size = min(actual_batch_size, 300)
        self.batch_size = actual_batch_size
        self.random_state = random_state

        dataset = self.full_dataset.sample(n=actual_batch_size, random_state=random_state)
        self.NormDataset = dataset.to_numpy(dtype=np.float64)
        self.n = len(self.NormDataset)
        self.m = self.NormDataset.shape[1]
        self.cols = list(dataset.columns)
        self.q = q
        self.device = device
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.labels = labels or []

        full_data = self.full_dataset.to_numpy(dtype=np.float64)
        self.RootMin = np.min(full_data, axis=0).reshape(1, -1)
        self.RootMax = np.max(full_data, axis=0).reshape(1, -1)
        self.Root = np.mean(full_data, axis=0).reshape(1, -1)

        label_mask = self._create_label_mask(labels)
        self.k = min(max_k, self.n) if max_k is not None else self.n
        if routing_profile in ('cp-routing-v1','cp-joint-v1','cp-v9','concept-graph-v1'):
            from .supervised import target_groups
            self._supervised_groups = target_groups(self)
            self._regression_indices=[self.cols.index(label) for label in self.labels if label in self.cols]
            if routing_profile in ('cp-joint-v1','cp-v9','concept-graph-v1') and self.labels and not (self._supervised_groups or self._regression_indices):
                raise ValueError('Target assente dalla codifica dei dati.')
            self.k = min(self.k, 256 if routing_profile in ('cp-joint-v1','cp-v9','concept-graph-v1') else 32)
            device = "numpy"

        numpy_class, torch_class = _AnalyticalClusterNet, ModularClusterNet
        if routing_profile == 'paper-routing-v1':
            from .paper_routing import PaperRoutingNumpy
            numpy_class = PaperRoutingNumpy
            if HAS_TORCH:
                from .paper_routing import PaperRoutingTorch
                torch_class = PaperRoutingTorch
        if HAS_TORCH and device != "numpy":
            self._use_torch = True
            self.X = torch.tensor(self.NormDataset, dtype=torch.float32, device=device)
            self.model_net = torch_class(
                self.n, self.m, k=self.k, device=device, tau=tau, alpha=alpha, label_mask=label_mask
            )
        else:
            self._use_torch = False
            self.X = self.NormDataset
            self.model_net = numpy_class(
                self.n,
                self.m,
                k=self.k,
                device="cpu",
                tau=tau,
                alpha=alpha,
                label_mask=label_mask,
                random_state=random_state,
            )

        self.model_net.initialize_G_from_data(self.X)
        self.model_net.set_root_statistics(self.Root, self.RootMin, self.RootMax)

        self.Transformation = np.eye(self.k, self.n)
        self.iteration = 0
        self._should_stop = False
        self.history = []
        self.Groups = None
        self.Relevance = None
        self.Counts = None
        self.Freqs = None
        self.cp = 0.0
        self.k_active = self.k

    def _create_label_mask(self, labels):
        if labels is None or len(labels) == 0:
            return None
        mask = np.zeros(self.m, dtype=bool)
        for idx, col in enumerate(self.cols):
            for lbl in labels:
                if col == lbl or col.startswith(str(lbl) + "_"):
                    mask[idx] = True
                    break
        return mask

    def _sample_new_batch(self, epoch):
        if self.total_n > self.batch_size:
            batch_df = self.full_dataset.sample(n=self.batch_size, random_state=self.random_state + epoch)
            self.NormDataset = batch_df.to_numpy(dtype=np.float64)
            self.n = len(self.NormDataset)
            if self._use_torch:
                self.X = torch.tensor(self.NormDataset, dtype=torch.float32, device=self.device)
            else:
                self.X = self.NormDataset

    def forward_pass(self):
        results = self.model_net(self.X)
        if self._use_torch:
            self.Transformation = results["T"].detach().cpu().numpy()
            self.Transformation_hard = results["T_hard"].detach().cpu().numpy()
            self.Groups = results["G"].detach().cpu().numpy()
            self.Relevance = results["R"].detach().cpu().numpy()
            self.Counts = results["C"].detach().cpu().numpy()
            self.Freqs = results["F"].detach().cpu().numpy()
            self.cp = float(results["cp"].item())
            self.k_active = int((results["T_hard"].sum(dim=1) > 0).sum().item())
        else:
            self.Transformation = results["T"].copy()
            self.Transformation_hard = results["T_hard"].copy()
            self.Groups = results["G"].copy()
            self.Relevance = results["R"].copy()
            self.Counts = results["C"].copy()
            self.Freqs = results["F"].copy()
            self.cp = float(results["cp"])
            self.k_active = int((results["T_hard"].sum(axis=1) > 0).sum())
        return results

    def nLearn(self, max_epochs=120, patience=25, min_delta=1e-4, log_interval=0):
        """
        Differentiable training loop optimizing Cluster Probability (cp) via Adam
        with automatic early stopping.
        """
        if self.routing_profile in ("cp-v9", "concept-graph-v1"):
            from .concept_adapter import train
            return train(self, max_epochs)
        if self.routing_profile == "cp-joint-v1":
            from .supervised import train
            return train(self, max_epochs)
        if self.routing_profile == "cp-routing-v1":
            from .variational import train
            return train(self, max_epochs)
        if self._use_torch:
            optimizer = optim.Adam(self.model_net.parameters(), lr=self.lr)
        else:
            optimizer = _NumpyAdam(self.model_net.matrices, lr=self.lr)

        best_cp = -np.inf
        best_state = None
        best_data = None
        wait = 0

        for epoch in range(max_epochs):
            if self._should_stop:
                break
            self.iteration = epoch
            self._sample_new_batch(epoch)
            # The reported objective belongs to the pre-update weights and this
            # exact batch. Preserve both as one candidate checkpoint.
            candidate_state = ({k: v.detach().clone() for k,v in self.model_net.matrices.items()}
                               if self._use_torch else
                               {k: v.copy() for k,v in self.model_net.matrices.items()})

            if self._use_torch:
                optimizer.zero_grad()
                results = self.model_net(self.X)
                loss = results["neg_cp"]
                loss.backward()
                optimizer.step()
                curr_cp = float(results["cp"].item())
                k_act = int((results["T_hard"].sum(dim=1) > 0).sum().item())
            else:
                results = self.model_net.forward(self.X)
                grads = self.model_net.backward()
                optimizer.step(grads)
                curr_cp = float(results["cp"])
                k_act = int((results["T_hard"].sum(axis=1) > 0).sum())

            self.history.append({"epoch": epoch, "cp": curr_cp, "loss": -curr_cp, "k_active": k_act})

            if curr_cp > best_cp + min_delta:
                best_cp = curr_cp
                wait = 0
                best_state = candidate_state
                best_data = self.NormDataset.copy()
            else:
                wait += 1
                if wait >= patience and epoch >= 20:
                    break

            if log_interval > 0 and epoch % log_interval == 0:
                print(f"[Epoch {epoch:03d}] cp={curr_cp:.5f}, k_active={k_act}")

        if best_state is not None:
            self.NormDataset = best_data
            self.n = len(best_data)
            self.X = (torch.tensor(best_data,dtype=self.X.dtype,device=self.device)
                      if self._use_torch else best_data)
            if self._use_torch:
                with torch.no_grad():
                    for k, v in best_state.items():
                        self.model_net.matrices[k].copy_(v)
            else:
                for k, v in best_state.items():
                    self.model_net.matrices[k] = v.copy()

        self.forward_pass()
        self.setModel()
        return self.model

    def setModel(self):
        self.model = AlgebraicClusterModel.summarize(
            pd.DataFrame(self.NormDataset, columns=self.cols)
        )
        transformation_row_mask = self.Transformation_hard.sum(axis=1) > 0
        if self.routing_profile in ("cp-routing-v1", "cp-joint-v1", "cp-v9", "concept-graph-v1"):
            transformation_row_mask[:] = True
        if not np.any(transformation_row_mask):
            transformation_row_mask[0] = True

        filtered_transformation = self.Transformation[transformation_row_mask].copy()
        filtered_transformation[filtered_transformation <= 0.05] = 0.0
        self.model.Transformation = filtered_transformation
        self.model.Groups = self.Groups[transformation_row_mask]
        self.model.Relevance = self.Relevance[transformation_row_mask]
        self.model.Counts = self.Counts[transformation_row_mask]
        self.model.cp = self.cp
        self.model.routing_profile = self.routing_profile
        if self.routing_profile in ('cp-joint-v1', 'cp-v9','concept-graph-v1'):
            self.model.Root = self.Root.copy()
            self.model.RootMin, self.model.RootMax = self.RootMin.copy(), self.RootMax.copy()
            self.model.RootError = self.full_dataset.to_numpy(dtype=float).std(axis=0,keepdims=True)
        if self.routing_profile == 'paper-routing-v1':
            self.model.routing_parameters = dict(alpha=self.model_net.relevance_alpha,
                beta=self.model_net.beta, gamma=self.model_net.mass_gamma)
        if self.routing_profile in ("cp-routing-v1", "cp-joint-v1", "cp-v9", "concept-graph-v1"):
            self.model.routing_parameters = self.variational_state
        self.model.Freqs = self.Freqs.T[transformation_row_mask]
        self.model.n = int(max(self.model.Counts.sum(), len(self.NormDataset)))
        return self.model

    # Threading compatibility methods for Dash UI
    _learning_thread = None
    _stop_event = None

    def start_nLearn_thread(self, max_epochs=500):
        self._stop_event = threading.Event()
        self._learning_thread = threading.Thread(target=self.nLearn, kwargs={"max_epochs": max_epochs, "patience": 80})
        self._learning_thread.daemon = True
        self._learning_thread.start()
        return self._learning_thread

    def is_nLearn_running(self):
        return self._learning_thread is not None and self._learning_thread.is_alive()

    def request_stop(self):
        self._should_stop = True
        return True

    def get_current_state(self):
        return {
            "running": self.is_nLearn_running(),
            "clusters": getattr(self, "k_active", 0),
            "cp": getattr(self, "cp", 0.0),
            "iteration": getattr(self, "iteration", 0),
            "n": getattr(self, "n", 0),
            "m": getattr(self, "m", 0),
        }

    def get_current_model(self):
        if self.routing_profile in ("cp-routing-v1", "cp-joint-v1", "cp-v9", "concept-graph-v1") and hasattr(self,"model"):
            return self.model
        self.forward_pass()
        return self.setModel()

    @staticmethod
    def gLearn(
        dataset,
        labels=[],
        gcm=None,
        similarity_threshold=0.92,
        current_depth=0,
        max_depth=2,
        min_samples_split=12,
        max_epochs=100,
        lr=0.08,
        tau=0.20,
        alpha=0.5,
        device="cpu",
        routing_profile="legacy-v8",
    ):
        """
        Recursive Hierarchical Concept Discovery (upgraded from v7 stub to full
        multi-level concept graph induction).
        """
        acl = AlgebraicClusterLearner(
            dataset, labels=labels, lr=lr, tau=tau, alpha=alpha, device=device, routing_profile=routing_profile
        )
        node = acl.nLearn(max_epochs=max_epochs)
        node.n = len(dataset)
        node.history = acl.history

        if gcm is not None:
            gcm.clusterModel[node.id] = node
            gcm.clusterData[node.id] = dataset

        # cp-v9 chooses its own level of generalization: a child is learned only on request.
        if routing_profile not in ('cp-v9','concept-graph-v1') and current_depth + 1 < max_depth and node.k() > 1 and len(dataset) >= min_samples_split:
            splits = node.splitDataset(dataset)
            full_lbls = node.fullLabels(labels)
            for k_idx, (sub_df, _) in splits.items():
                if len(sub_df) < min_samples_split or len(sub_df) >= len(dataset):
                    continue
                # Check label impurity if labels are provided, or feature variance otherwise
                if full_lbls:
                    if routing_profile == 'cp-joint-v1':
                        # Learned class readouts can be confident even when a
                        # branch contains mixed labels. Split on observed
                        # training purity, never on a fitted readout or test.
                        max_class_prob = min((float(sub_df.iloc[:,indices].mean().max())
                                             for indices in acl._supervised_groups),default=1.)
                    else:
                        max_class_prob = float(np.max(node.Groups[k_idx, [node.cols.index(c) for c in full_lbls]]))
                    needs_split = max_class_prob < similarity_threshold
                    if routing_profile=='cp-joint-v1' and acl._regression_indices:
                        needs_split |= float(np.mean(np.std(sub_df.iloc[:,acl._regression_indices].to_numpy(),axis=0)))>.15
                else:
                    needs_split = float(np.mean(np.std(sub_df.to_numpy(), axis=0))) > 0.15

                if needs_split:
                    _, child_node = AlgebraicClusterLearner.gLearn(
                        sub_df,
                        labels=labels,
                        gcm=gcm,
                        similarity_threshold=similarity_threshold,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        max_epochs=max(40, max_epochs // 2),
                        lr=lr,
                        tau=tau,
                        alpha=alpha,
                        device=device,
                        routing_profile=routing_profile,
                    )
                    node.children[k_idx] = child_node.id

        return gcm, node

    @staticmethod
    def learn(
        dataset,
        labels=[],
        max_depth=1,
        max_epochs=100,
        lr=0.08,
        tau=0.20,
        alpha=0.5,
        device="cpu",
        routing_profile="legacy-v8",
    ):
        trainDataset, minMax = GraphClusterModel.normalizeData(
            GraphClusterModel.full_dummies(dataset)
        )
        gcm = GraphClusterModel()
        gcm.minMax = minMax
        gcm.labels = labels

        gcm, root = AlgebraicClusterLearner.gLearn(
            trainDataset,
            labels=labels,
            gcm=gcm,
            max_depth=max_depth,
            max_epochs=max_epochs,
            lr=lr,
            tau=tau,
            alpha=alpha,
            device=device,
            routing_profile=routing_profile,
        )
        gcm.rootId = root.id
        gcm.clusterModel[root.id] = root
        gcm.clusterModel = dict(sorted(gcm.clusterModel.items()))
        gcm.clusterData[root.id] = trainDataset
        return gcm


class ACANPredictor:
    """
    Scikit-learn compatible White-Box Concept Attention Estimator for classification,
    regression, and Universal Masked Tabular Autocompletion.
    """

    def __init__(
        self,
        max_depth=1,
        max_epochs=100,
        lr=0.08,
        tau=0.18,
        alpha=0.6,
        use_soft_attention=True,
        device="cpu",
        routing_profile="legacy-v8",
    ):
        self.max_depth = max_depth
        self.max_epochs = max_epochs
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.routing_profile = routing_profile
        self.use_soft_attention = use_soft_attention
        self.device = device
        self.gcm = None
        self.target_name = "target"

    def fit(self, df_or_X, y=None, target_cols=None):
        if isinstance(df_or_X, pd.DataFrame) and target_cols is not None:
            train_df = df_or_X.copy()
            self.labels = list(target_cols)
        else:
            if isinstance(df_or_X, pd.DataFrame):
                train_df = df_or_X.copy()
            else:
                train_df = pd.DataFrame(df_or_X, columns=[f"f{i}" for i in range(df_or_X.shape[1])])
            if y is not None:
                train_df[self.target_name] = [str(v) for v in y]
                self.labels = [self.target_name]
            else:
                self.labels = []

        self.gcm = AlgebraicClusterLearner.learn(
            train_df,
            labels=self.labels,
            max_depth=self.max_depth,
            max_epochs=self.max_epochs,
            lr=self.lr,
            tau=self.tau,
            alpha=self.alpha,
            device=self.device,
            routing_profile=self.routing_profile,
        )
        return self

    def predict(self, X_df):
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=[f"f{i}" for i in range(X_df.shape[1])])
        Y_pred, _, _ = self.gcm.predictDataset(X_df, self.labels, use_soft_attention=self.use_soft_attention)
        root = self.gcm.clusterModel[self.gcm.rootId]
        full_lbls = root.fullLabels(self.labels)
        if len(full_lbls) > 1 and len(self.labels) == 1:
            prefix = str(self.labels[0]) + "_"
            preds = [col[len(prefix) :] if col.startswith(prefix) else col for col in Y_pred.idxmax(axis=1)]
            return np.array(preds)
        return Y_pred.to_numpy()

    def complete(self, incomplete_df):
        return self.gcm.predict_masked(incomplete_df)

    def explain_concepts(self, top_features=4):
        root = self.gcm.clusterModel[self.gcm.rootId]
        return root.extractRules(minMax=self.gcm.minMax, top_features=top_features)
