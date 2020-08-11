import numpy as np
from typing import List
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, OutlierMixin, ClassifierMixin
from sklearn.neighbors._base import SupervisedIntegerMixin, UnsupervisedMixin
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Hypercube:
    coord: tuple
    idxs: List[int] = field(default_factory=list)
    ngb_density: int = 0

class DensityStrategy:
    """It defines an interface to implement the strategy
    used to efficiently compute the neighborhood density of hypercubes."""

    def __init__(self, H, n_jobs, verbose):
        self.H = H # local reference
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.w_max = 0 # max neighborhood density

    def get_densities(self):
        """Returns the neighborhood density for each hypercube."""
        return []

    def get_max_density(self):
        """Returns the maximum neighborhood density value."""
        return self.w_max

    def _is_immediate(self, hi: Hypercube, hk: Hypercube):
        for p, q in zip(hi.coord, hk.coord):
            if abs(p - q) > 1:
                return False
        return True

    def _is_prospective(self, hi: Hypercube, hk: Hypercube, j: int):
        return abs(hi.coord[j] - hk.coord[j]) <= 1

class NaiveStrategy(DensityStrategy):
    """ Perform a linear search over the existing hypercubes
        to compute the neighborhood density for each hypercube."""
    def __init__(self, H, n_jobs, verbose):
        super().__init__(H, n_jobs, verbose)

    def _get_density(self, i, n):
        self.H[i].ngb_density = len(self.H[i].idxs)

        k = i - 1
        while k >= 0:
            if not self._is_prospective(self.H[i], self.H[k], 0):
                break
            if self._is_immediate(self.H[i], self.H[k]):
                self.H[i].ngb_density += len(self.H[k].idxs)
            k -= 1

        k = i + 1
        while k < n:
            if not self._is_prospective(self.H[i], self.H[k], 0):
                break
            if self._is_immediate(self.H[i], self.H[k]):
                self.H[i].ngb_density += len(self.H[k].idxs)
            k += 1

        return self.H[i].ngb_density

    def get_densities(self):
        n = len(self.H)

        W = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')\
                    (delayed(self._get_density)(i, n) for i in range(n))

        for i in range(n):
            self.H[i].ngb_density = W[i]
            self.w_max = max(self.w_max, W[i])

        return self.H

class TreeStrategy(DensityStrategy):
    """ Perform a tree-based search over the existing hypercubes
        to compute the neighborhood density for each hypercube."""

    class Node:
        def __init__(self, value, begin, end):
            self.value = value
            self.begin = begin
            self.end = end
            self.childs = {}
        
        def add(self, node):
            if node: self.childs[node.value] = node

    def __init__(self, min_split, H, n_jobs, verbose):
        super().__init__(H, n_jobs, verbose)
        self.num_total_rows = len(self.H) - 1
        self.num_total_cols = len(self.H[0].coord) - 1
        self.min_split = min(max(1, min_split), self.num_total_rows)
        self.root = self.Node(-1, 0, self.num_total_rows)

        self._build_index(self.root, 0)
    
    def _build_index(self, parent, col):
        # stop sub-mapping when the parent node map less than min_split hypercubes
        if (parent.end - parent.begin) < self.min_split:
            return

        # initialize the next range
        value = self.H[parent.begin].coord[col]
        begin = parent.begin
        end = -1
        
        i = parent.begin
        while i <= parent.end:

            # when the value change a node is created
            if self.H[i].coord[col] != value:
                # mark the end of the current value 
                end = i - 1
                # create node for 'value' in 'col'
                child = self.Node(value, begin, end)
                parent.add(child)
                # map child values in the next dimension
                self._build_index(child, col + 1)
                # start new range
                begin = i
                # update value
                value = self.H[i].coord[col]

            i += 1
        
        # map last value
        end = i - 1
        child = self.Node(value, begin, end)
        parent.add(child)

        self._build_index(child, col + 1)

    def _get_density(self, hi, parent, col):
        density = 0

        # when there is no childs, scan over the parent range
        if not parent.childs:
            k = parent.begin
            while k <= parent.end:
                hk = self.H[k]
                if self._is_immediate(hi, hk):
                    density += len(self.H[k].idxs)
                k += 1
        else:
            # when there are childs we must recursevly find the
            # nodes that store value of 1 unit distant
            # from the current value
            lft_val = hi.coord[col] - 1
            mid_val = hi.coord[col]
            rgt_val = hi.coord[col] + 1

            lft_node = parent.childs.get(lft_val)
            mid_node = parent.childs.get(mid_val)
            rgt_node = parent.childs.get(rgt_val)

            next_col = min(col + 1, self.num_total_cols)

            if lft_node: density += self._get_density(hi, lft_node, next_col)
            if mid_node: density += self._get_density(hi, mid_node, next_col)
            if rgt_node: density += self._get_density(hi, rgt_node, next_col)

        return density

    def get_densities(self):
        n = len(self.H)

        W = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')\
                    (delayed(self._get_density)(self.H[i], self.root, 0) for i in range(n))

        for i in range(n):
            self.H[i].ngb_density = W[i]
            self.w_max = max(self.w_max, W[i])

        return self.H


class HySortOD(BaseEstimator, SupervisedIntegerMixin, 
               UnsupervisedMixin, ClassifierMixin,
               OutlierMixin):
    """Perform HySortOD outlier detection from vector array.
    Detect outliers based on the neighborhood notion, where instances
    with few neighbors are very likely outlying instances. It reports
    an outlierness score for each instance measuring the degree of
    being an outlier.

    Parameters
    ----------
    num_bins : int, optional (default=5)
        The number of partitions to be considered in each dimension.
        It dictates the hypercube granularity, length and neighborhood radius.

    strategy_name : {"naive", "tree"}, optional (default='tree')
        The strategy to use when searching the hypercube neighborhood density.
        - 'naive': perform a linear search (recommended for small datasets)
        - 'tree': perform a tree-based search

    min_split : int, optional (default=100)
        The number of hypercubes to be mapped by the tree-node.
        Only used if `strategy_name=tree`. `1` is equivalent to `strategy_name=naive`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel. `-1` means using all processors.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    scores_ : ndarray, shape (n_samples, )
        Outlierness scores for each point in the dataset given to fit().

    References
    ----------
    .. [1] Cabral, E., Cordeiro, R. (2020, October).
       Fast and Scalable Outlier Detection with Sorted Hypercubes.
       In International Conference on Information and Knowledge Management
       (pp. XXX-XXX). ACM.
    """
    def __init__(self, num_bins=5, strategy_name='tree', min_split=100, n_jobs=1, verbose=0):
        self.num_bins = num_bins
        self.strategy_name = strategy_name
        self.min_split = min_split
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.length = 1 / float(num_bins)
        self.scaler = MinMaxScaler()
        self.H = [] # sorted-hypercubes list
        self.scores_ = []

    def _more_tags(self):
        return {
            'allow_nan': False,
            'binary_only': True,
            'requires_fit': True
        }
    
    def _to_hypercube(self, instance):
        """ Convert instance to hypercube coordinates. """
        return tuple(np.floor(instance / self.length).astype(int))
    
    def _search(self, coord):
        """ Perform binary search in sorted hypercubes list. """
        lo, hi = 0, len(self.H) - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if self.H[mid].coord < coord:
                lo = mid + 1
            elif coord < self.H[mid].coord:
                hi = mid - 1
            else:
                return mid, True
        return lo, False

    def _get_strategy_instance(self):
        if self.strategy_name == 'naive':
            return NaiveStrategy(self.H, self.n_jobs, self.verbose)
        return TreeStrategy(self.min_split,self.H, self.n_jobs, self.verbose)

    def _get_neighborhood_densities(self):
        """ Compute the neighborhood density for all hypercubes
            using the specified strategy. """
        self.strategy = self._get_strategy_instance()
        self.H = self.strategy.get_densities()

    def _get_sorted_hypercubes(self, X):
        """ Build and sort hypercubes for dataset X. """
        for idx, instance in enumerate(X):
            coord = self._to_hypercube(instance)
            pos, exists = self._search(coord)
            if exists:
                self.H[pos].idxs.append(idx)
            else:
                self.H.insert(pos, Hypercube(coord, [idx], 0))

    def _score(self, h):
        """ Calculate the score relative to the highest density value. """
        w_max = float(self.strategy.get_max_density())
        return 1 - (h.ngb_density / w_max)

    def _get_outlierness_scores(self, m):
        """ Compute outlierness scores for all instances in existing hypercubes. """
        self.scores_ = np.zeros(m)
        for h in self.H:
            for idx in h.idxs:
                self.scores_[idx] = self._score(h)

    def fit(self, X, y=None):
        """Fit the model using X as training data
        Parameters
        ----------
        X : array-like, Training data. If array or matrix, shape [n_samples, n_features].
        """
        self.scaler.fit(X)
        self._get_sorted_hypercubes(self.scaler.transform(X))
        self._get_neighborhood_densities()
        self._get_outlierness_scores(len(X))
        return self

    def predict(self, X):
        """
        Predict based on the training set.
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features) samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        y = []
        for instance in self.scaler.transform(X):
            coord = self._to_hypercube(instance)
            pos, exists = self._search(coord)
            score = self._score(self.H[pos]) if exists else 1.0
            y.append(score)
        return y

    def fit_predict(self, X, y=None):
        """Perform fit on X and returns labels for X.
        Return values between 0 and 1, where outliers instances
        has values close to 1 and inliers close to 0.
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
            (n_samples, n_features)
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Inliers receive scores close to 0, outliers receive scores close to 1.
        """
        return self.fit(X).predict(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the ROC AUC Score on the given test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            ROC AUC Score of self.predict(X) wrt. y.
        """
        return roc_auc_score(y, self.predict(X), sample_weight=sample_weight)

    