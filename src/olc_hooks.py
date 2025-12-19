import torch


def make_block_projectors(d: int, C: int, device, dtype):
    assert d % C == 0
    dc = d // C
    Ps = []
    for c in range(C):
        P = torch.zeros(d, d, device=device, dtype=dtype)
        P[c * dc : (c + 1) * dc, c * dc : (c + 1) * dc] = torch.eye(
            dc, device=device, dtype=dtype
        )
        Ps.append(P)
    return Ps


class OLCController:
    def __init__(self, d_model: int, num_channels: int, device, dtype):
        self.d_model = d_model
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.Ps = make_block_projectors(d_model, num_channels, device, dtype)
        self.channel = 0

    def set_channel(self, c: int):
        self.channel = c

    def project(self, x: torch.Tensor):
        # Ensure projector matches runtime dtype/device (important when HF uses autocast / device_map)
        P = self.Ps[self.channel]
        if P.device != x.device or P.dtype != x.dtype:
            P = P.to(device=x.device, dtype=x.dtype)
            self.Ps[self.channel] = P
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
