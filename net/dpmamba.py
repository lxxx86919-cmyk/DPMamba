import torch
import torch.nn as nn

from net.ITA import TNet, TBNet
from net.mamba import JNet_mamba
from net.modules import LightweightPhysicsBranch
from net.physics import LearnablePhysicsHead, manual_A_torch, physical_reconstruction


class DPMambaUIE(nn.Module):
    """
    Switch-controlled DPMamba-UIE model for ablation experiments.

    Outputs:
        J, t, tb, A, I_rec, aux
    """
    def __init__(
        self,
        use_dynamic_mamba=False,
        use_learnable_physics=False,
        use_frequency_module=False,
        use_lightweight=False,
        use_a_residual_map=False,
        a_residual_scale=0.1,
        light_base_channels=32,
    ):
        super(DPMambaUIE, self).__init__()
        self.use_dynamic_mamba = bool(use_dynamic_mamba)
        self.use_learnable_physics = bool(use_learnable_physics)
        self.use_frequency_module = bool(use_frequency_module)
        self.use_lightweight = bool(use_lightweight)
        self.use_a_residual_map = bool(use_a_residual_map)
        self.a_residual_scale = a_residual_scale

        self.Jnet = JNet_mamba(
            use_dynamic_mamba=self.use_dynamic_mamba,
            use_frequency_module=self.use_frequency_module,
        )

        if self.use_lightweight:
            self.light_physics = LightweightPhysicsBranch(
                in_channels=3,
                base_channels=light_base_channels,
                predict_a=self.use_learnable_physics,
                use_a_residual_map=self.use_a_residual_map,
                a_residual_scale=self.a_residual_scale,
            )
            self.Tnet = None
            self.TBnet = None
        else:
            self.light_physics = None
            self.Tnet = TNet()
            self.TBnet = TBNet()

        if self.use_learnable_physics and not self.use_lightweight:
            self.physics_head = LearnablePhysicsHead(
                in_channels=3,
                hidden_dim=32,
                use_a_residual_map=self.use_a_residual_map,
                a_residual_scale=self.a_residual_scale,
            )
        else:
            self.physics_head = None

    def _collect_scan_weights(self):
        if not self.use_dynamic_mamba:
            return None
        return {
            'mamba1': self.Jnet.mamba1.last_scan_weights,
            'mamba2': self.Jnet.mamba2.last_scan_weights,
            'mamba3': self.Jnet.mamba3.last_scan_weights,
        }

    def forward(self, x):
        J = self.Jnet(x)

        light_aux = None
        physics_aux = None
        if self.use_lightweight:
            t, tb, light_A, light_aux = self.light_physics(x)
        else:
            t = self.Tnet(x)
            tb = self.TBnet(x)
            light_A = None

        if self.use_learnable_physics:
            if self.use_lightweight:
                A = light_A
                physics_aux = light_aux
            else:
                A, physics_aux = self.physics_head(x)
        else:
            A = manual_A_torch(x)

        I_rec = physical_reconstruction(J, t, tb, A)

        aux = {
            'scan_weights': self._collect_scan_weights(),
            'frequency_features': None,
            'A_global': None,
            'A_map': A,
            'lightweight_features': None,
            'used_dynamic_mamba': self.use_dynamic_mamba,
            'used_learnable_physics': self.use_learnable_physics,
            'used_frequency_module': self.use_frequency_module,
            'used_lightweight': self.use_lightweight,
        }

        if physics_aux is not None:
            aux['A_global'] = physics_aux.get('A_global')
            aux['A_map'] = physics_aux.get('A_map', A)
            aux['A_residual'] = physics_aux.get('A_residual')
        else:
            aux['A_residual'] = None

        if light_aux is not None:
            aux['lightweight_features'] = light_aux.get('features')
            if aux['A_global'] is None:
                aux['A_global'] = light_aux.get('A_global')

        return J, t, tb, A, I_rec, aux
