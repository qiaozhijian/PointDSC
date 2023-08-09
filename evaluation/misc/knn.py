# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import numpy as np
from scipy.spatial import cKDTree
import time
from misc.metrics import pdist


def torch_intersect(Na, Nb, i_ab, j_ab, i_ba, j_ba):
  def make_sparse_mat(i_, j_, sz):
    inds = torch.cat([i_.reshape([1, -1]),
                      j_.reshape([1, -1])], dim=0)
    vals = torch.ones_like(inds[0, :])

    M = torch.sparse.FloatTensor(inds, vals, sz)
    return M

  sz = [Na, Nb]
  M_ab = make_sparse_mat(i_ab, j_ab, sz)
  M_ba = make_sparse_mat(i_ba, j_ba, sz)

  M = M_ab.add(M_ba).coalesce()
  i, j = M._indices()
  v = M._values()
  is_both = (v == 2)
  i_final = i[is_both]
  j_final = j[is_both]

  return i_final, j_final


def nn_to_mutual(feats0, feats1, corres_idx0, corres_idx1, idx1_2nd=None, force_return_2nd=False):
  uniq_inds_1 = torch.unique(corres_idx1)
  inv_corres_idx1, inv_corres_idx0, _ = find_nn(feats1[uniq_inds_1, :], feats0, False)
  inv_corres_idx1 = uniq_inds_1

  final_corres_idx0, final_corres_idx1 = torch_intersect(
    feats0.shape[0], feats1.shape[0],
    corres_idx0, corres_idx1,
    inv_corres_idx0, inv_corres_idx1)

  if idx1_2nd is not None:
    idx1_2nd = idx1_2nd[
      final_corres_idx0]  # this relies on the fact that corres_idx0 is the full sorted range [0,...,n]
    return final_corres_idx0, final_corres_idx1, idx1_2nd
  elif force_return_2nd:
    return final_corres_idx0, final_corres_idx1, None
  else:
    return final_corres_idx0, final_corres_idx1


def find_nn(F0, F1, return_2nd=False):
  nn_max_n = 250

  def knn_dist(f0, f1):
    # Fast implementation with torch.einsum()
    with torch.no_grad():
      # L2 distance:
      dist2 = torch.sum(f0 ** 2, dim=1).reshape([-1, 1]) + torch.sum(f1 ** 2, dim=1).reshape(
        [1, -1]) - 2 * torch.einsum('ac,bc->ab', f0, f1)
      dist = dist2.clamp_min(1e-30).sqrt_()
      # Cosine distance:
      #   dist = 1-torch.einsum('ac,bc->ab', f0, f1)

      _, ind = dist.min(dim=1, keepdim=True)
      if return_2nd:
        dist[torch.arange(len(ind)), ind.squeeze()] = np.inf
        _, ind_2nd = dist.min(dim=1, keepdim=True)

        return ind, ind_2nd
      else:
        return ind, None

  N = len(F0)
  C = int(np.ceil(N / nn_max_n))
  stride = nn_max_n
  inds = []
  inds_2nd = []
  for i in range(C):
    with torch.no_grad():
      ind, ind_2nd = knn_dist(F0[i * stride:(i + 1) * stride], F1)
      inds.append(ind)
      inds_2nd.append(ind_2nd)

  inds = torch.cat(inds)
  if return_2nd:
    inds_2nd = torch.cat(inds_2nd)
  assert len(inds) == N

  corres_idx0 = torch.arange(len(inds)).long().squeeze()
  corres_idx1 = inds.long().squeeze().cpu()
  if return_2nd:
    idx1_2nd = inds_2nd.long().squeeze().cpu()
    return corres_idx0, corres_idx1, idx1_2nd
  else:
    return corres_idx0, corres_idx1, None


def find_2nn(fcgf_feats0, fcgf_feats1):
  # 1st nearest neighbor are already available, we're only re-calculating them here for
  # convenience, therefore we don't measure the time for this calculation. However,
  # 2nd nearest neighbors are not available, and we do want to take into consideration
  # the time that it takes to calculate them. This allow fair comparison with algorithms
  # tha don't need them,. such as PointDSC.
  simple_corres_start_time = time.time()
  _, _, _ = find_nn(fcgf_feats0, fcgf_feats1, return_2nd=False)
  simple_corres_time = time.time() - simple_corres_start_time
  corres_start_time = time.time()
  corres_idx0, corres_idx1, idx1_2nd = find_nn(fcgf_feats0, fcgf_feats1, return_2nd=True)
  corres_time = time.time() - corres_start_time
  additional_time_for_finding_2nd_closest = corres_time - simple_corres_time
  return corres_idx0, corres_idx1, idx1_2nd, additional_time_for_finding_2nd_closest

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds


def find_knn_gpu(F0, F1, nn_max_n=-1, knn=1, return_distance=False):

  def knn_dist(f0, f1, knn=1, dist_type='L2'):
    knn_dists, knn_inds = [], []
    with torch.no_grad():
      dist = pdist(f0, f1, dist_type=dist_type)
      min_dist, ind = dist.min(dim=1, keepdim=True)

      knn_dists.append(min_dist)
      knn_inds.append(ind)

      if knn > 1:
        for k in range(knn - 1):
          NR, NC = dist.shape
          flat_ind = (torch.arange(NR) * NC).type_as(ind) + ind.squeeze()
          dist.view(-1)[flat_ind] = np.inf
          min_dist, ind = dist.min(dim=1, keepdim=True)

          knn_dists.append(min_dist)
          knn_inds.append(ind)

    min_dist = torch.cat(knn_dists, 1)
    ind = torch.cat(knn_inds, 1)

    return min_dist, ind

  # Too much memory if F0 or F1 large. Divide the F0
  if nn_max_n > 1:
    N = len(F0)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    dists, inds = [], []

    for i in range(C):
      with torch.no_grad():
        dist, ind = knn_dist(F0[i * stride:(i + 1) * stride], F1, knn=knn, dist_type='L2')
      dists.append(dist)
      inds.append(ind)

    dists = torch.cat(dists)
    inds = torch.cat(inds)
    assert len(inds) == N

  else:
    dist = pdist(F0, F1, dist_type='SquareL2')
    min_dist, inds = dist.min(dim=1)
    dists = min_dist.detach().unsqueeze(1) #.cpu()
    # inds = inds.cpu()
  if return_distance:
    return inds, dists
  else:
    return inds


def find_knn_batch(F0,
                   F1,
                   len_batch,
                   return_distance=False,
                   nn_max_n=-1,
                   knn=1,
                   search_method=None,
                   concat_results=False):
  if search_method is None or search_method == 'gpu':
    return find_knn_gpu_batch(
        F0,
        F1,
        len_batch=len_batch,
        nn_max_n=nn_max_n,
        knn=knn,
        return_distance=return_distance,
        concat_results=concat_results)
  elif search_method == 'cpu':
    return find_knn_cpu_batch(
        F0,
        F1,
        len_batch=len_batch,
        knn=knn,
        return_distance=return_distance,
        concat_results=concat_results)
  else:
    raise ValueError(f'Search method {search_method} not defined')


def find_knn_gpu_batch(F0,
                       F1,
                       len_batch,
                       nn_max_n=-1,
                       knn=1,
                       return_distance=False,
                       concat_results=False):
  dists, nns = [], []
  start0, start1 = 0, 0
  for N0, N1 in len_batch:
    nn = find_knn_gpu(
        F0[start0:start0 + N0],
        F1[start1:start1 + N1],
        nn_max_n=nn_max_n,
        knn=knn,
        return_distance=return_distance)
    if return_distance:
      nn, dist = nn
      dists.append(dist)
    if concat_results:
      nns.append(nn + start1)
    else:
      nns.append(nn)
    start0 += N0
    start1 += N1

  if concat_results:
    nns = torch.cat(nns)
    if return_distance:
      dists = torch.cat(dists)

  if return_distance:
    return nns, dists
  else:
    return nns


def find_knn_cpu_batch(F0,
                       F1,
                       len_batch,
                       knn=1,
                       return_distance=False,
                       concat_results=False):
  if not isinstance(F0, np.ndarray):
    F0 = F0.detach().cpu().numpy()
    F1 = F1.detach().cpu().numpy()

  dists, nns = [], []
  start0, start1 = 0, 0
  for N0, N1 in len_batch:
    nn = find_knn_cpu(
        F0[start0:start0 + N0], F1[start1:start1 + N1], return_distance=return_distance)
    if return_distance:
      nn, dist = nn
      dists.append(dist)
    if concat_results:
      nns.append(nn + start1)
    else:
      nns.append(nn + start1)
    start0 += N0
    start1 += N1

  if concat_results:
    nns = np.hstack(nns)
    if return_distance:
      dists = np.hstack(dists)

  if return_distance:
    return torch.from_numpy(nns), torch.from_numpy(dists)
  else:
    return torch.from_numpy(nns)
