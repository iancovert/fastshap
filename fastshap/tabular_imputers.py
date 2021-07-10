import torch
import torch.nn as nn
import numpy as np


class BaselineImputer:
    '''
    Evaluate a model while replacing features with baseline values.

    Args:
      model: predictive model (e.g., torch.nn.Module).
      baseline: baseline values.
      groups: (optional) feature groups, represented by a list of lists.
    '''

    def __init__(self, model, baseline, groups=None, link=None):
        # Store model.
        self.model = model

        # Store baseline.
        device = next(model.parameters()).device
        if isinstance(baseline, np.ndarray):
            baseline = torch.tensor(baseline, dtype=torch.float32,
                                    device=device)
        elif isinstance(baseline, torch.Tensor):
            baseline = baseline.to(device=device)
        else:
            raise ValueError('baseline must be np.ndarray or torch.Tensor')
        self.baseline = baseline

        # Set up link.
        if link is None:
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Store feature groups.
        num_features = baseline.shape[1]
        if groups is None:
            self.num_players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.num_players = len(groups)
            self.groups_matrix = torch.zeros(
                len(groups), num_features, dtype=torch.float32, device=device)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = 1

    def __call__(self, x, S):
        '''
        Evaluate model with baseline values.

        Args:
          x: input examples.
          S: coalitions.
        '''
        # Prepare x and S.
        device = next(self.model.parameters()).device
        if isinstance(x, np.ndarray):
            numpy_conversion = True
            x = torch.tensor(x, dtype=torch.float32, device=device)
            S = torch.tensor(S, dtype=torch.float32, device=device)
        else:
            numpy_conversion = False
        if self.groups_matrix is not None:
            S = torch.mm(S, self.groups_matrix)

        # Replace specified indices.
        x_baseline = S * x + (1 - S) * self.baseline

        # Make predictions.
        pred = self.link(self.model(x_baseline))
        if numpy_conversion:
            pred = pred.cpu().data.numpy()
        return pred


class MarginalImputer:
    '''
    Evaluate a model while replacing features with samples from the marginal
    distribution.

    Args:
      model: predictive model (e.g., torch.nn.Module).
      background: background values representing the marginal distribution.
      groups: (optional) feature groups, represented by a list of lists.
    '''

    def __init__(self, model, background, groups=None, link=None):
        # Store model.
        self.model = model

        # Store background samples.
        device = next(self.model.parameters()).device
        if isinstance(background, np.ndarray):
            background = torch.tensor(background, dtype=torch.float32,
                                      device=device)
        elif isinstance(background, torch.Tensor):
            background = background.to(device=device)
        else:
            raise ValueError('background must be np.ndarray or torch.Tensor')
        self.background = background
        self.background_repeat = background
        self.n_background = len(background)

        # Set up link.
        if link is None:
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Store feature groups.
        num_features = background.shape[1]
        if groups is None:
            self.num_players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.num_players = len(groups)
            self.groups_matrix = torch.zeros(
                len(groups), num_features, dtype=torch.float32, device=device)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = 1

    def __call__(self, x, S):
        '''
        Evaluate model with marginal imputation.

        Args:
          x: input examples.
          S: coalitions.
        '''
        # Prepare x and S.
        device = next(self.model.parameters()).device
        if isinstance(x, np.ndarray):
            numpy_conversion = True
            x = torch.tensor(x, dtype=torch.float32, device=device)
            S = torch.tensor(S, dtype=torch.float32, device=device)
        else:
            numpy_conversion = False
        if self.groups_matrix is not None:
            S = torch.mm(S, self.groups_matrix)

        # Set up background repeat.
        if len(self.background_repeat) != len(x) * self.n_background:
            self.background_repeat = self.background.repeat(len(x), 1)

        # Prepare x and S.
        x_tiled = x.unsqueeze(1).repeat(1, self.n_background, 1).reshape(
            len(x) * self.n_background, -1)
        S_tiled = S.unsqueeze(1).repeat(1, self.n_background, 1).reshape(
            len(x) * self.n_background, -1)

        # Replace features.
        x_tiled = S_tiled * x_tiled + (1 - S_tiled) * self.background_repeat

        # Make predictions.
        pred = self.link(self.model(x_tiled))
        pred = pred.reshape(len(x), self.n_background, *pred.shape[1:])
        pred = torch.mean(pred, dim=1)
        if numpy_conversion:
            pred = pred.cpu().data.numpy()
        return pred
