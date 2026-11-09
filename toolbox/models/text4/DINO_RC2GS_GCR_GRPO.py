"""RC2GS with component-wise Scale-Causal GRPO.

The segmentation network, Gaussian fields, route policy and inference path are
inherited unchanged from DINO_RC2GS_SAR. This file only adds no-gradient,
single-scale counterfactual rollouts required to assign a separate causal
credit to every reward component and each of the four route decisions.

Keeping the original class/file name preserves the project's existing model
factory and configuration interface while avoiding another experimental file.
"""

import torch
import torch.nn.functional as F

from .DINO_RC2GS_SAR import EncoderDecoder as StableRC2GSEncoderDecoder


class EncoderDecoder(StableRC2GSEncoderDecoder):
    """Stable RC2GS plus critic-free, component-causal training rollouts."""

    @torch.no_grad()
    def _credit_decode(self, features):
        """Decode detached features deterministically for reward measurement."""
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            return self.decoder([feature.detach() for feature in features])
        finally:
            self.decoder.train(was_training)

    def forward_group(
        self,
        rgb,
        depth,
        group_size=4,
        temperature=1.0,
        compute_counterfactual=True,
    ):
        if int(group_size) < 2:
            raise ValueError("Scale-Causal GRPO requires grpo_group_size >= 2")

        group_size = int(group_size)
        policy_feature, features = self._extract(rgb)
        routes, auxiliary_loss = self._experts(features, depth, rgb)

        # Use exactly the light GaussianRoutePolicy of the stable SAR model.
        # There is no Transformer and no separate behaviour policy.
        policy_logits = self.policy(policy_feature.detach(), depth, rgb)
        distribution = torch.distributions.Categorical(
            logits=policy_logits / max(float(temperature), 1e-4)
        )

        group_actions = []
        scale_log_probs = []
        selected_group = []
        packed_features = [[] for _ in routes]
        for _ in range(group_size):
            actions = distribution.sample()
            selected = self._select(routes, actions)
            selected_group.append(selected)
            for scale, feature in enumerate(selected):
                packed_features[scale].append(feature)
            group_actions.append(actions)
            scale_log_probs.append(distribution.log_prob(actions))

        packed_features = [torch.cat(items, dim=0) for items in packed_features]
        native_logits = self.decoder(packed_features)
        full_logits = F.interpolate(
            native_logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        group_logits = full_logits.reshape(
            group_size,
            rgb.shape[0],
            self.num_classes,
            rgb.shape[-2],
            rgb.shape[-1],
        )

        # The detached current-policy expectation is the neutral reference.
        # Intervening on one scale while holding the other three sampled routes
        # fixed yields a scale-specific difference reward without a critic.
        probability = policy_logits.softmax(dim=-1).detach()
        credit_logits = None
        counterfactual_logits = None
        if compute_counterfactual:
            reference_routes = []
            for scale, candidates in enumerate(routes):
                weight = probability[:, scale].reshape(-1, 3, 1, 1, 1)
                reference_routes.append(
                    (candidates.detach() * weight).sum(dim=1)
                )

            counterfactual_native = []
            for intervention_scale in range(len(routes)):
                counterfactual_features = [[] for _ in routes]
                for selected in selected_group:
                    for scale in range(len(routes)):
                        counterfactual_features[scale].append(
                            reference_routes[scale]
                            if scale == intervention_scale
                            else selected[scale]
                        )
                counterfactual_features = [
                    torch.cat(items, dim=0) for items in counterfactual_features
                ]
                counterfactual_native.append(
                    self._credit_decode(counterfactual_features)
                )

            credit_logits = self._credit_decode(packed_features)
            native_size = credit_logits.shape[-2:]
            credit_logits = credit_logits.reshape(
                group_size,
                rgb.shape[0],
                self.num_classes,
                native_size[0],
                native_size[1],
            )
            counterfactual_logits = torch.stack(
                counterfactual_native, dim=0
            ).reshape(
                len(routes),
                group_size,
                rgb.shape[0],
                self.num_classes,
                native_size[0],
                native_size[1],
            )

        # Preserve the stable supervised expected-mixture path exactly. The
        # detached probabilities prevent a hidden continuous policy shortcut.
        expected_features = []
        for scale, candidates in enumerate(routes):
            weight = probability[:, scale].reshape(-1, 3, 1, 1, 1)
            expected_features.append((candidates * weight).sum(dim=1))
        expected_logits = self._decode(expected_features, rgb.shape[-2:])

        scale_log_probs = torch.stack(scale_log_probs, dim=0)
        return {
            "logits": group_logits,
            "expected_logits": expected_logits,
            "log_probs": scale_log_probs.sum(dim=-1),
            "scale_log_probs": scale_log_probs,
            "actions": torch.stack(group_actions, dim=0),
            "entropy": distribution.entropy().sum(dim=1).mean(),
            "policy_logits": policy_logits,
            "credit_logits": credit_logits,
            "counterfactual_logits": counterfactual_logits,
            "auxiliary_loss": auxiliary_loss,
        }
