import torch
from utils import helpers
# Import the specific SAE loading function
from sae_cnn import load_sae_from_checkpoint

class BaseModelZeroAblationExperiment:
    """
    Handles zero-ablation interventions on base model activations for one or more channels.
    """
    def __init__(self, model_path, target_layer, device=None):
        """
        Initializes the experiment.

        Args:
            model_path (str): Path to the base model checkpoint.
            target_layer (str): Name of the layer to target (e.g., 'conv_seqs.2.res_block1.conv1').
            device (torch.device, optional): Device to run on. Defaults to CUDA if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BaseZeroAblation: Using device: {self.device}")

        print(f"BaseZeroAblation: Loading model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        self.model.eval()

        self.layer_name = target_layer
        self.module = self._get_module(self.layer_name)
        if self.module is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in the model.")
        print(f"BaseZeroAblation: Targeting layer: {self.layer_name}")

        self.channels_to_zero = None # List of channel indices to zero
        self.hook_handle = None

    def _get_module(self, layer_name):
        """Gets the module object from the model using its layer name."""
        for name, mod in self.model.named_modules():
            if name == layer_name:
                return mod
        return None

    def _zeroing_hook(self, module, input, output):
        """Hook function to zero out specified channels."""
        if self.channels_to_zero is not None and len(self.channels_to_zero) > 0:
            modified_output = output.clone()
            # Ensure channels_to_zero contains valid indices
            # Note: Assumes output shape is (batch, channels, height, width) or similar
            # For Linear layers, shape might be (batch, features) - adjust slicing if needed
            if output.ndim >= 2: # Basic check for channel dimension
                 num_actual_channels = output.shape[1]
                 valid_indices = [idx for idx in self.channels_to_zero if 0 <= idx < num_actual_channels]
                 if len(valid_indices) != len(self.channels_to_zero):
                      print(f"Warning: Invalid channel indices detected in {self.channels_to_zero} for layer with {num_actual_channels} channels. Using only valid indices: {valid_indices}")

                 if valid_indices:
                      # Zero out the specified channels
                      # Works for Conv layers (B, C, H, W) and potentially Linear (B, F) if dim 1 is features
                      modified_output[:, valid_indices] = 0.0
                 return modified_output
            else:
                 print(f"Warning: Output tensor dimension ({output.ndim}) not suitable for channel zeroing. Skipping.")
                 return output # Return original if shape is wrong
        # If no channels specified or hook inactive, return original output
        return output

    def set_channels_to_zero(self, channel_indices):
        """
        Specifies which channels to zero out and activates the intervention hook.

        Args:
            channel_indices (list[int]): A list of channel indices to zero.
                                         An empty list or None disables zeroing.
        """
        if not isinstance(channel_indices, list) and channel_indices is not None:
             raise TypeError("channel_indices must be a list of integers or None.")

        self.channels_to_zero = channel_indices

        # Register hook if not already active and channels are specified
        if self.hook_handle is None and self.channels_to_zero is not None and len(self.channels_to_zero) > 0:
            self.hook_handle = self.module.register_forward_hook(self._zeroing_hook)
            # print(f"Base Hook registered for channels: {self.channels_to_zero}") # Debug
        elif self.hook_handle is not None and (self.channels_to_zero is None or len(self.channels_to_zero) == 0):
             # Remove hook if no channels are specified anymore
             self.disable_zeroing()

    def disable_zeroing(self):
        """Removes the intervention hook and resets the channel list."""
        if self.hook_handle is not None:
            # print("Removing base hook.") # Debug
            self.hook_handle.remove()
            self.hook_handle = None
        self.channels_to_zero = None

    def get_target_layer(self):
        """Returns the PyTorch module object for the target layer."""
        return self.module

    def get_num_channels(self):
        """Attempts to determine the number of output channels/features for the target layer."""
        target_layer_module = self.module
        if target_layer_module is None:
             return None # Should have been caught in init

        if hasattr(target_layer_module, 'out_channels'):
            return target_layer_module.out_channels
        elif hasattr(target_layer_module, 'out_features'): # For Linear layers
             return target_layer_module.out_features
        else:
            # Try getting output shape by dummy forward pass
            try:
                # Need input shape. Assuming it's available via model's embedder
                if not hasattr(self.model, 'embedder') or not hasattr(self.model.embedder, 'observation_space'):
                     print("Warning: Cannot determine input shape for dummy pass.")
                     return None
                dummy_input_shape = self.model.embedder.observation_space.shape # (C, H, W)
                dummy_input = torch.zeros(1, *dummy_input_shape, device=self.device)
                temp_handle = None
                output_shape = None
                def temp_hook(module, input, output):
                    nonlocal output_shape
                    output_shape = output.shape
                temp_handle = target_layer_module.register_forward_hook(temp_hook)
                with torch.no_grad():
                    self.model(dummy_input)
                if temp_handle: temp_handle.remove()

                if output_shape and len(output_shape) > 1:
                    return output_shape[1] # Channels/Features are typically dim 1 (B, C, H, W) or (B, F)
                else:
                     print(f"Warning: Could not determine output channels via dummy pass. Output shape: {output_shape}")
                     return None
            except Exception as e:
                 print(f"Warning: Error during dummy forward pass for channel count: {e}")
                 return None


class SAEZeroAblationExperiment:
    """
    Handles zero-ablation interventions on SAE features corresponding to a base model layer.
    """
    def __init__(self, model_path, sae_checkpoint_path, layer_name, layer_number, device=None):
        """
        Initializes the experiment.

        Args:
            model_path (str): Path to the base model checkpoint.
            sae_checkpoint_path (str): Path to the SAE checkpoint (.pt file).
            layer_name (str): Name of the base model layer the SAE corresponds to.
            layer_number (int): Layer number (used for context/logging).
            device (torch.device, optional): Device to run on. Defaults to CUDA if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAEZeroAblation: Using device: {self.device}")

        # Load Base Model
        print(f"SAEZeroAblation: Loading base model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        self.model.eval()

        # Load SAE using the helper function
        print(f"SAEZeroAblation: Loading SAE from {sae_checkpoint_path}")
        try:
            # Use the helper function to load the SAE
            self.sae = load_sae_from_checkpoint(sae_checkpoint_path).to(self.device)
            self.sae.eval() # Ensure SAE is in evaluation mode
            sae_dim = self.get_num_features() # Get dimension after loading
            if sae_dim is None:
                 raise ValueError("Could not determine SAE dimension after loading.")
            print(f"SAEZeroAblation: Loaded SAE with {sae_dim} features.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAE using load_sae_from_checkpoint from {sae_checkpoint_path}: {e}")

        self.layer_name = layer_name
        self.layer_number = layer_number # Store for reference
        self.module = self._get_module(self.layer_name) # Hook the base model layer
        if self.module is None:
            raise ValueError(f"Base model layer '{self.layer_name}' not found.")
        print(f"SAEZeroAblation: Targeting base layer: {self.layer_name} for SAE intervention")

        self.features_to_zero = None # List of feature indices to zero
        self.hook_handle = None

    def _get_module(self, layer_name):
        """Gets the module object from the model using its layer name."""
        for name, mod in self.model.named_modules():
            if name == layer_name:
                return mod
        return None

    def _sae_zeroing_hook(self, module, input, output):
        """
        Hook function placed on the base model layer.
        It encodes the base layer's output, zeros specified SAE features,
        decodes, and returns the reconstructed activations.
        """
        if self.features_to_zero is not None and len(self.features_to_zero) > 0:
            with torch.no_grad():
                sae_features = self.sae.encode(output)
                modified_sae_features = sae_features.clone()

                # --- Zero out specified features ---
                if sae_features.ndim >= 2:
                     num_actual_features = sae_features.shape[1]
                     valid_indices = [idx for idx in self.features_to_zero if 0 <= idx < num_actual_features]
                     if len(valid_indices) != len(self.features_to_zero):
                          print(f"Warning: Invalid SAE feature indices detected in {self.features_to_zero} for SAE with {num_actual_features} features. Using only valid indices: {valid_indices}")

                     if valid_indices:
                          # Zero out the specified feature activations
                          modified_sae_features[:, valid_indices] = 0.0
                else:
                     print(f"Warning: SAE feature tensor dimension ({sae_features.ndim}) not suitable for feature zeroing. Skipping.")
                     modified_sae_features = sae_features # Use original if shape is wrong

                reconstructed_activations = self.sae.decode(modified_sae_features)
                return reconstructed_activations
        # If no features specified or hook inactive, return original output
        return output

    def set_features_to_zero(self, feature_indices):
        """
        Specifies which SAE features to zero out and activates the intervention hook
        on the corresponding base model layer.

        Args:
            feature_indices (list[int]): A list of SAE feature indices to zero.
                                         An empty list or None disables zeroing.
        """
        if not isinstance(feature_indices, list) and feature_indices is not None:
             raise TypeError("feature_indices must be a list of integers or None.")

        self.features_to_zero = feature_indices

        # Register hook on the base model layer if not already active
        if self.hook_handle is None and self.features_to_zero is not None and len(self.features_to_zero) > 0:
            self.hook_handle = self.module.register_forward_hook(self._sae_zeroing_hook)
            # print(f"SAE Hook registered for features: {self.features_to_zero}") # Debug
        elif self.hook_handle is not None and (self.features_to_zero is None or len(self.features_to_zero) == 0):
             self.disable_zeroing()


    def disable_zeroing(self):
        """Removes the intervention hook and resets the feature list."""
        if self.hook_handle is not None:
            # print("Removing SAE hook.") # Debug
            self.hook_handle.remove()
            self.hook_handle = None
        self.features_to_zero = None

    def get_target_layer(self):
        """Returns the PyTorch module object for the base model target layer."""
        return self.module

    def get_num_features(self):
        """Returns the number of features (dimensionality) of the loaded SAE."""
        # Prioritize config attribute if it exists (common pattern)
        if hasattr(self.sae, 'cfg') and hasattr(self.sae.cfg, 'd_sae'):
            return self.sae.cfg.d_sae
        # Fallback to common attribute names
        elif hasattr(self.sae, 'sae_dim'):
             return self.sae.sae_dim
        elif hasattr(self.sae, 'd_sae'):
             return self.sae.d_sae
        # Fallback to inferring from weights (less reliable)
        elif hasattr(self.sae, 'W_enc'):
             # Shape is typically (activation_dim, sae_dim)
             return self.sae.W_enc.shape[1]
        elif hasattr(self.sae, 'encoder') and hasattr(self.sae.encoder, 'weight'):
             # Shape is typically (sae_dim, activation_dim) for nn.Linear
             return self.sae.encoder.weight.shape[0]
        else:
             print("Warning: Could not determine number of SAE features from known attributes (cfg.d_sae, sae_dim, d_sae, W_enc, encoder.weight).")
             return None