import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import RCP_analysis as rcp

def _ipca_template_per_channel(signal, template_old, rank):
    """
    Use incremental PCA to fix one channel
    
    Input:
    signal: n_stim x time
    template_old: rank x time
    
    Returns:
    A dictionary for one channel and the artifact corrected signal
    """
    ipca = IncrementalPCA(n_components=rank, batch_size=1)
    signal_ipca = ipca.fit_transform(signal)
    template = template_old
    
    return signal, template
    
def ipca_all(X, rank):
    """
    Use incremental PCA to fix all channels
    
    Returns dictionary for all channels (ex: 128) and the artifact corrected signals for all channels
    """
    n_stim, n_time, n_channels = X.shape
    templates_all = np.zeros(n_stim, rank, n_channels)
    signal_all = np.zeros(1, rank, n_channels)
    for ch in range(n_channels):
        signal = X[ch, :]
        signal_corr, template = _ipca_template_per_channel(signal, template, rank)
        templates_all[ch] = template
        signal_all[ch] = signal_corr
    return signal_all, templates_all