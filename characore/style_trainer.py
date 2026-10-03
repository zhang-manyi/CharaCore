"""GRPOTrainer subclass that drops rejected groups from the update instead of zero-filling them.

Verified against TRL 0.26.2 grpo_trainer.py: _calculate_rewards maps None to NaN
(line 1204), _generate_and_score_completions sums with nansum (1940) and takes
plain group mean/std, so a whole-None group arrives with zero rewards and zero
advantages. Zero advantage alone still leaves the KL term in the loss (added at
2223, before the mask at 2225), so the group's completion_mask is zeroed as well:
those rows then contribute nothing to the policy gradient or the KL. They still
count in the per-sequence .mean() denominator of loss_type="grpo", which scales
that step's loss by kept/total; the count is logged so this is visible.

The override runs before _prepare_inputs shuffles and splits the generation
batch, so rows are still consecutive groups of num_generations. Single process
only: with DDP the reward sees all ranks' rows but each rank holds a slice.
"""


def apply_rejection(output, rejected, group_size):
    """Zero completion_mask and advantages for every row of each rejected group, in place."""
    rows = output["completion_mask"].shape[0]
    if rows % group_size:
        raise ValueError("generation batch is not whole groups")
    for group in rejected:
        if not 0 <= group < rows // group_size:
            raise ValueError(f"rejected group {group} outside this batch")
        span = slice(group * group_size, (group + 1) * group_size)
        output["completion_mask"][span] = 0
        output["advantages"][span] = 0
    return output


def make_trainer_class():
    from trl import GRPOTrainer

    class StyleGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, style_reward, **kwargs):
            super().__init__(*args, **kwargs)
            if self.accelerator.num_processes != 1:
                raise ValueError("group rejection is verified for a single process only")
            self.style_reward = style_reward
            self.rejected_total = 0

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            rejected = self.style_reward.take_rejected()
            apply_rejection(output, rejected, self.num_generations)
            self.rejected_total += len(rejected)
            mode = "train" if self.model.training else "eval"
            metrics = self.style_reward.last_metrics
            self._metrics[mode]["style/rejected_groups"].append(len(rejected))
            for key in ("hard_violation_rate", "style_score_mean", "length_mean", "marker_rate",
                        "win_rate_vs_base", "judge_calls", "reused_requests", "varied_groups"):
                if metrics.get(key) is not None:
                    self._metrics[mode][f"style/{key}"].append(float(metrics[key]))
            return output

    return StyleGRPOTrainer
