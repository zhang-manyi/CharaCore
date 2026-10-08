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
batch, so rows are still consecutive groups of num_generations. Under DDP the
reward sees all ranks' rows concatenated in rank order (gather at 1223) while
each rank keeps rows [rank * n, (rank + 1) * n) (process_slice at 1967); the
reward returns global group indices and each rank masks the part it holds.
"""


def apply_rejection(output, rejected, group_size, offset=0, total=None):
    """Zero completion_mask and advantages, in place, for this rank's rows of each rejected group.

    output holds global rows [offset, offset + rows) of a batch of total rows (default: all of it).
    """
    rows = output["completion_mask"].shape[0]
    total = rows if total is None else total
    if total % group_size:
        raise ValueError("generation batch is not whole groups")
    if not 0 <= offset <= total - rows:
        raise ValueError("local rows fall outside the batch")
    for group in rejected:
        if not 0 <= group < total // group_size:
            raise ValueError(f"rejected group {group} outside this batch")
        start, end = max(group * group_size, offset), min((group + 1) * group_size, offset + rows)
        if start < end:
            span = slice(start - offset, end - offset)
            output["completion_mask"][span] = 0
            output["advantages"][span] = 0
    return output


def make_trainer_class():
    from trl import GRPOTrainer

    class StyleGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, style_reward, **kwargs):
            super().__init__(*args, **kwargs)
            self.style_reward = style_reward
            self.rejected_total = 0

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            rejected = self.style_reward.take_rejected()
            rows = output["completion_mask"].shape[0]
            apply_rejection(output, rejected, self.num_generations, offset=self.accelerator.process_index * rows,
                            total=rows * self.accelerator.num_processes)
            self.rejected_total += len(rejected)
            mode = "train" if self.model.training else "eval"
            metrics = self.style_reward.last_metrics
            self._metrics[mode]["style/rejected_groups"].append(len(rejected))
            for key in ("hard_violation_rate", "style_score_mean", "length_mean", "marker_rate",
                        "asuka_benxiaojie_open_rate", "stock_opener_rate", "reward_mean", "margin_mean",
                        "margin_positive_rate", "policy_prn_mean", "base_prn_mean", "identical_rate",
                        "win_rate_vs_base", "order_inconsistent_rate",
                        "judge_calls", "reused_requests", "varied_groups"):
                if metrics.get(key) is not None:
                    self._metrics[mode][f"style/{key}"].append(float(metrics[key]))
            return output

    return StyleGRPOTrainer
