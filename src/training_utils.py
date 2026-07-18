from contextlib import contextmanager

import torch as th


class ModelEMA:
    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError('EMA decay must be between 0 and 1, got {}'.format(decay))
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: th.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @th.no_grad()
    def update(self, model):
        self.num_updates += 1
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                raise ValueError('EMA is missing trainable parameter {}'.format(name))
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model):
        if self.num_updates == 0:
            yield
            return

        backup = {}
        correction = 1.0 - self.decay ** self.num_updates
        with th.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name]).div_(correction)
        try:
            yield
        finally:
            with th.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(backup[name])


def supervision_loss_weights(flag_path, smooth_ccal):
    if not flag_path:
        return 0.0, 0.0
    if smooth_ccal:
        return 1.0 / 3.0, 1.0 / 6.0
    return 1.0, 0.5


def replace_case_50_with_case_0(graph_info):
    """Replace case ID 50 with case ID 0 while preserving the target case ID."""
    cases = graph_info['delay-label_pairs']
    case_indices = graph_info.get('case_indices')
    if case_indices is not None:
        case_positions = {case_index: position for position, case_index in enumerate(case_indices)}
        source_position = case_positions.get(0)
        target_position = case_positions.get(50)
        if source_position is not None and target_position is not None:
            cases[target_position] = cases[source_position]
        return

    if len(cases) > 50:
        cases[50] = cases[0]


def normalize_endpoint_correlation(weights):
    """Normalize each endpoint's node-correlation mass without producing NaNs."""
    mass = weights.sum(dim=1, keepdim=True)
    return weights / mass.clamp_min(1e-12)


def valid_endpoint_mask(labels):
    """Return a row mask for endpoints with an available nonnegative label."""
    if not th.is_tensor(labels):
        raise TypeError('labels must be a torch tensor')
    if labels.ndim == 0:
        return (labels >= 0).reshape(1)
    return (labels >= 0).reshape(labels.shape[0], -1).all(dim=1)


def filter_endpoint_rows(value, mask):
    """Filter an endpoint-aligned tensor, leaving scalars/sentinels unchanged."""
    if value is None or not th.is_tensor(value) or value.ndim == 0:
        return value
    if value.shape[0] != mask.shape[0]:
        return value
    return value[mask]
