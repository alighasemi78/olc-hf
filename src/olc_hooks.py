import torch


def make_block_projectors(d: int, C: int, device):
    assert d % C == 0
    dc = d // C
    Ps = []
    for c in range(C):
        P = torch.zeros(d, d, device=device)
        P[c * dc : (c + 1) * dc, c * dc : (c + 1) * dc] = torch.eye(dc, device=device)
        Ps.append(P)
    return Ps


class OLCController:
    """
    Controls which channel is active for the current forward pass.
    Call set_channel(i) before each agent step.
    """

    def __init__(self, d_model: int, num_channels: int, device):
        self.Ps = make_block_projectors(d_model, num_channels, device)
        self.channel = 0

    def set_channel(self, c: int):
        self.channel = c

    def project(self, x: torch.Tensor):
        # x: [B, T, d] or [T, d] depending on model internals; handle last dim
        P = self.Ps[self.channel]
        return x @ P.T


def attach_projection_hooks(model, controller: OLCController):
    """
    Approximation:
    - After attention output: project
    - After MLP output: project
    This depends on HF model module names; we attach generically to common patterns.
    """
    handles = []

    def post_hook(_module, _inputs, output):
        # output might be tuple in some architectures
        if isinstance(output, tuple):
            out0 = output[0]
            out0p = controller.project(out0)
            return (out0p,) + output[1:]
        return controller.project(output)

    for name, module in model.named_modules():
        lname = name.lower()
        # Common patterns:
        # - attention output projection: "attn.o_proj" or "self_attn.o_proj"
        # - MLP output projection: "mlp.down_proj" or "mlp.c_proj"
        if any(k in lname for k in ["self_attn.o_proj", "attn.o_proj", "o_proj"]):
            handles.append(module.register_forward_hook(post_hook))
        if (
            any(
                k in lname
                for k in ["mlp.down_proj", "mlp.c_proj", "down_proj", "c_proj"]
            )
            and "attn" not in lname
        ):
            handles.append(module.register_forward_hook(post_hook))

    return handles
