import numpy as np
from sklearn.decomposition import IncrementalPCA
import RCP_analysis as rcp
from spikeinterface.core import BaseRecording, BaseRecordingSegment


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

class _PerChannelIPCACorrectedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, micro_map_per_channel, micro_corrected):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.micro_map_per_channel = micro_map_per_channel  # {ch_idx: [(start, end), ...]}
        self.micro_corrected = micro_corrected
        
        # Build lookup structures per channel for fast searching
        self.channel_starts = {}
        self.channel_ends = {}
        
        for ch_idx, map_list in micro_map_per_channel.items():
            if map_list:
                starts = np.array([s for s, e in map_list])
                ends = np.array([e for s, e in map_list])
                order = np.argsort(starts)
                self.channel_starts[ch_idx] = starts[order]
                self.channel_ends[ch_idx] = ends[order]
            else:
                self.channel_starts[ch_idx] = np.array([], dtype=np.int64)
                self.channel_ends[ch_idx] = np.array([], dtype=np.int64)

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()
        
        # Handle different channel_indices types
        if channel_indices is None:
            channel_indices = np.arange(len(self.micro_map_per_channel))
        elif isinstance(channel_indices, slice):
            # Convert slice to array
            start = channel_indices.start if channel_indices.start is not None else 0
            stop = channel_indices.stop if channel_indices.stop is not None else len(self.micro_map_per_channel)
            step = channel_indices.step if channel_indices.step is not None else 1
            channel_indices = np.arange(start, stop, step)
        else:
            channel_indices = np.asarray(channel_indices)
        
        # Process each channel independently
        for local_idx, global_ch_idx in enumerate(channel_indices):
            if global_ch_idx not in self.channel_starts:
                continue
            
            starts = self.channel_starts[global_ch_idx]
            ends = self.channel_ends[global_ch_idx]
            
            if len(starts) == 0:
                continue
            
            # Find pulses that overlap this request
            lo = np.searchsorted(ends, start_frame, side='right')
            hi = np.searchsorted(starts, end_frame, side='left')
            
            for pulse_idx in range(lo, hi):
                p_start = starts[pulse_idx]
                p_end = ends[pulse_idx]
                
                overlap_start = max(start_frame, p_start)
                overlap_end = min(end_frame, p_end)
                
                if overlap_start < overlap_end:
                    chunk_idx_start = overlap_start - start_frame
                    chunk_idx_end = overlap_end - start_frame
                    patch_idx_start = overlap_start - p_start
                    patch_idx_end = overlap_end - p_start
                    
                    # Extract this channel's corrected patch
                    patch = self.micro_corrected[pulse_idx, patch_idx_start:patch_idx_end, global_ch_idx]
                    
                    # Place it in the output
                    traces[chunk_idx_start:chunk_idx_end, local_idx] = patch
        
        return traces

class PerChannelIPCACorrectedRecording(BaseRecording):
    def __init__(self, parent_recording, micro_map_per_channel, micro_corrected):
        BaseRecording.__init__(self, 
                               parent_recording.get_sampling_frequency(), 
                               parent_recording.channel_ids, 
                               parent_recording.get_dtype())
        self.parent_recording = parent_recording
        parent_recording.copy_metadata(self)
        
        for segment_index in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[segment_index]
            self.add_recording_segment(
                _PerChannelIPCACorrectedRecordingSegment(parent_segment, micro_map_per_channel, micro_corrected)
            )



"""
To run this code:

corrector = IPCA_Artifact_Correction(rank = 3)
signal_corrected, templates = corrector.ipca_all(signal)

    where signal is a n_stim * n_time * n_channels array
    and templates is a list of Template objects for each channel
        - for example if we wanted to apply the template to a new signal without adjusting the weights again
"""