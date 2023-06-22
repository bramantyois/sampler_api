import numpy as np
from app.sampler.sampler import sample


def test_sampler():
    n_dim_to_keep = np.random.randint(1, 10)
    n_hist_artists = np.random.randint(11, 20)
    def get_non_zero_weights(weights):
        return weights[weights > 0]

    # test none weights
    weights = sample(n_hist_artists, weights=None, n_dim_to_keep=n_dim_to_keep)
    assert get_non_zero_weights(weights).shape[-1] == n_dim_to_keep

    # test if weights shape shoudl be (1,-1)
    assert weights.shape[0] == 1
    assert weights.shape[1] == n_hist_artists

    # test weight having more than n_dim_to_keep non-zero dimensions
    weights = np.ones(n_hist_artists)
    new_weights = sample(
        n_hist_artists=n_hist_artists, 
        weights=weights, 
        n_dim_to_keep=n_dim_to_keep)

    assert get_non_zero_weights(new_weights).shape[-1] == n_dim_to_keep

    # test if weights are always positive
    assert np.all(new_weights >= 0)

    # test if nwe weights is different
    weights = new_weights

    assert weights.shape[1] == n_hist_artists

    new_weights = sample(
        n_hist_artists=n_hist_artists, 
        weights=weights, 
        n_dim_to_keep=n_dim_to_keep)
    assert not np.all(weights == new_weights)

    # test if weights shape shoudl be (1,-1)
    assert new_weights.shape[0] == 1
    assert new_weights.shape[1] == n_hist_artists