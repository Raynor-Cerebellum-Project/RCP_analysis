import numpy as np
from sklearn.decomposition import IncrementalPCA
import RCP_analysis as rcp


class Template:
    """
    Store IPCA weights for each channel
    """
    def __init__(self, weights=None):
        self.weights = weights
    
    def __getitem__(self, index):
        return self.weights[index] if self.weights is not None else None
    
    def __setitem__(self, index, value):
        if self.weights is not None:
            self.weights[index] = value
    
    def __len__(self):
        return len(self.weights) if self.weights is not None else 0
    
    def __repr__(self):
        return f"IPCA_Template({self.weights})"
        
    def update_weights(self, new_weights, learning_rate=0.5):
        if self.weights is None:
            self.weights = new_weights
        else:
            self.weights = (1 - learning_rate) * self.weights + learning_rate * new_weights


class IPCA_Artifact_Correction:
    """
    Artifact correction using Incremental PCA (from sklearn.decomposition)
    """
    def __init__(self, rank = 3):
        self.rank = rank
        

    def ipca_template_per_channel(self, signal, template, learning_rate=0.9):
        """
        Incremental PCA to correct one channel
        
        Input:
            signal: n_stim * n_time array
            template: rank * n_time array
        
        Returns:
            signal_corrected: artifact-corrected channel
            template: updated template with new weights
        """
        n_stim, n_time = signal.shape

        baseline_mean = np.mean(signal[:, :3], axis=1, keepdims=True)
        signal = signal - baseline_mean
        
        effective_rank = min(self.rank, n_time, n_stim)
        ipca = IncrementalPCA(n_components=effective_rank)

        ipca.partial_fit(signal)
        
        V_new = ipca.components_
        template.update_weights(V_new, learning_rate=learning_rate)
        
        artifact = signal @ template.weights.T @ template.weights
        signal_corrected = signal - artifact + baseline_mean
        
        return signal_corrected, template

    def ipca_all(self, signal):
        """
        Incremental PCA to correct all channels.
        
        Input:
            signal: n_stim * n_time * n_channels array
        
        Returns:
            signal_all: artifact-corrected signals (n_stim * n_time * n_channels)
            templates_all: list of Template objects for each channel
        """
        n_stim, n_time, n_channels = signal.shape
        
        signal_all = np.zeros_like(signal)
        templates_all = []
        
        for ch in range(n_channels):
            signal_ch = signal[:, :, ch]
            
            template = Template()
            signal_corr, template = self.ipca_template_per_channel(signal_ch, template)

            signal_all[:, :, ch] = signal_corr
            templates_all.append(template)
            
        return signal_all, templates_all

    def apply_template(self, signal, template):
        """
        Apply existing template to new signal WITHOUT updating the weight!
        
        Input:
            signal: n_stim * n_time array (raw, un-centered)
            template: Template object with existing weights
        
        Returns:
            signal_corrected: artifact-corrected signal (with baseline preserved)
        """
        baseline_mean = np.mean(signal[:, :3], axis=1, keepdims=True)

        centered = signal - baseline_mean
        X_artifact = centered @ template.weights.T @ template.weights


        # print(signal[0, :], np.mean(signal[0, :3]))
        # print(centered[0, :], np.mean(centered[0, :3]))
        # print(X_artifact[0, :], np.mean(X_artifact[0, :3]))
        # print(centered[0, :] - X_artifact[0, :], np.mean(centered[0, :] - X_artifact[0, :]) )
        # print(centered[0, :] - X_artifact[0, :] + baseline_mean[0, :], np.mean(centered[0, :] - X_artifact[0, :] + baseline_mean[0, :]) )
        return centered - X_artifact + baseline_mean



"""
To run this code:

corrector = IPCA_Artifact_Correction(rank = 3)
signal_corrected, templates = corrector.ipca_all(signal)

    where signal is a n_stim * n_time * n_channels array
    and templates is a list of Template objects for each channel
        - for example if we wanted to apply the template to a new signal without adjusting the weights again
"""