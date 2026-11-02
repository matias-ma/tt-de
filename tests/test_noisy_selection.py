import numpy as np
import pytest

from ftde.noisy_model_selection import select_by_paired_noisy_nll


def test_paired_selection_ignores_common_log_density_variation():
    common = np.array([-100., 100., -200., 200.])
    logs = np.stack([common, common-.1])
    chosen, detail = select_by_paired_noisy_nll(logs, [2., 1.])
    assert chosen == 0  # Second model is consistently worse, despite huge raw NLL SE.
    assert detail['candidates'][1]['paired_cluster_se'] < 1e-13
    shifted, _ = select_by_paired_noisy_nll(logs-common, [2., 1.])
    assert shifted == chosen


def test_paired_selection_prefers_smooth_model_when_difference_is_uncertain():
    logs = np.stack([np.zeros(4), np.array([-.4, .3, -.4, .3])])
    chosen, detail = select_by_paired_noisy_nll(logs, [2., 1.])
    assert chosen == 1
    assert detail['minimum_nll_index'] == 0


def test_cluster_standard_error_preserves_within_chain_dependence():
    differences = np.tile([-.2, .4], 10)
    logs = np.stack([np.zeros(20), -differences])
    _, independent = select_by_paired_noisy_nll(logs, [2., 1.])
    _, grouped = select_by_paired_noisy_nll(logs, [2., 1.], cluster_ids=np.arange(20) % 2)
    np.testing.assert_allclose(grouped['candidates'][1]['paired_cluster_se'], .3)
    assert grouped['candidates'][1]['paired_cluster_se'] > independent['candidates'][1]['paired_cluster_se']
    with pytest.raises(ValueError, match='two independent clusters'):
        select_by_paired_noisy_nll(logs, [2., 1.], cluster_ids=np.zeros(20))
    with pytest.raises(ValueError, match='finite'):
        select_by_paired_noisy_nll([[0., np.nan]], [1.])
