import numpy as np
from sklearn.decomposition import IncrementalPCA
import RCP_analysis as rcp


class Template:
    """
    Store IPCA weights for each channel
    """
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else []
    
    def __getitem__(self, index):
        return self.weights[index]
    
    def __setitem__(self, index, value):
        self.weights[index] = value
    
    def __len__(self):
        return len(self.weights)
    
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
        

    def ipca_template_per_channel(self, signal, template, learning_rate=0.5):
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
        
        ipca = IncrementalPCA(n_components=self.rank)

        for i in range(n_stim):
            stim_data = signal[i:i+1, :]  # Keep 2D shape
            ipca.partial_fit(stim_data)
        
        V_new = ipca.components_
        template._weights(V_new, learning_rate=0.5)
        
        artifact = signal @ template.weights.T @ template.weights
        signal_corrected = signal - artifact
        
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
            signal_corr, template = self.ipca_template_per_channel(signal, template)

            signal_all[:, :, ch] = signal_corr
            templates_all.append(template)
            
        return signal_all, templates_all
    
    def apply_template(self, signal, template):
        """
        Apply existing template to new signal WITHOUT updating the weight!
        
        Input:
            signal: n_stim * n_time array
            template: Template object with existing weights
        
        Returns:
            signal_corrected: artifact-corrected signal
        """
        V = template.weights
        X_artifact = signal @ V.T @ V
        return signal - X_artifact



"""
To run this code:

corrector = IPCA_Artifact_Correction(rank = 3)
signal_corrected, templates = corrector.ipca_all(signal)

    where signal is a n_stim * n_time * n_channels array
    and templates is a list of Template objects for each channel
        - for example if we wanted to apply the template to a new signal without adjusting the weights again
"""