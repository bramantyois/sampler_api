import numpy as np
from typing import List, Optional, Union


# sampler
def sample(
    n_hist_artists: int,
    weights: Union[None, np.ndarray] = None, 
    std: float = 0.01, 
    n_dim_to_keep: int = 8,)->np.ndarray:
    """
    Samples a vector of weights from a dirichlet distribution. 
    If weights is not None, then it mutates the weights by adding noise to the non-zero dimensions. 
    Return weights should always be >=0.

    Args:
        n_hist_artists (int): number of historical artists
        weights (Union[None, np.ndarray], optional): vector of weights. Defaults to None.
        std (float, optional): standard deviation of the noise. Defaults to 0.01.
        n_dim_to_keep (int, optional): number of non-zero dimensions to keep. Defaults to 8.

    Returns:
        np.ndarray: vector of weights of the size (1, n_hist_artists)
    """
    
    # check if weights should be freshly sampled
    should_sample_new = False
    if weights is None:
        should_sample_new=True
    else:
        # raise error if weights length is not equal to embeddings length
        if weights.shape[-1] != n_hist_artists:
            #logger.error("weights length is not equal to embeddings length")
            raise ValueError("weights length is not equal to embeddings length")

        # get mask of non-zero dimensions
        weights = np.array(weights)
        non_zero_weights = weights[weights > 0]

        # check if the number of non-zero dimensions is not equal to n_dim_to_keep
        if non_zero_weights.shape[-1] != n_dim_to_keep:
            should_sample_new = True

    # now sample new weights or mutate the existing weights
    if should_sample_new: # generate new weights
        #logger.warning("the number of non-zero dimensions is not equal to n_dim_to_keep. Sampling from dirichlet distribution")
        non_zero_weights = np.random.dirichlet(np.ones(n_dim_to_keep), size=1)[0]
        # choose random index 
        rand_idx = np.random.choice(np.arange(n_hist_artists), size=n_dim_to_keep, replace=False)

        weights = np.zeros(n_hist_artists)
        weights[rand_idx] = non_zero_weights

        return weights.reshape(1, -1)
    else: # mutate weights
        # get mask of non-zero dimensions
        mask = weights > 0
        non_zero_weights = weights[weights > 0]

        # normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # add noise to the non-zero weights
        non_zero_weights += np.random.normal(0, scale=std, size=(non_zero_weights.shape))
        neg_mask = non_zero_weights < 0

        num_neg = np.sum(neg_mask)
        neg_weights = non_zero_weights[neg_mask]

        # if there are negative dimensions, set them to zero
        non_zero_weights[neg_mask] = 0

        weights[mask] = non_zero_weights

        # get indices of zero dimensions
        new_mask = weights == 0
        mask_idx = np.where(new_mask)[0]
        rand_idx = np.random.choice(mask_idx, size=num_neg, replace=False)
    
        # add the negative weights to the random dimensions
        weights[0,rand_idx] = -neg_weights

        # normalize
        weights /= np.sum(weights)

        return weights.reshape(1, -1)
