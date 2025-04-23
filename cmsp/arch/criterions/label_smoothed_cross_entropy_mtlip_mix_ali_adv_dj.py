import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    use_mix: bool = field(
        default=False,
        metadata={"help:": "use mix loss"},
    )
    use_adv: bool = field(
        default=False,
        metadata={"help:": "use adv loss"},
    )


# 判别器模型（包含3个全连接层和1个输出层）
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        # 三个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)

        # 输出层，进行语音与文本的二分类
        self.fc_out = nn.Linear(hidden_dim // 4, 1)  # 二分类（0.2或0.8）
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)  # 输出层
        return x


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_mtlip_mix_ali_adv_dj", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            use_mix=False,
            use_adv=False,
            mix_prob=0.2,
            ot_type="L2",
            ip_dim=512,
            op_dim=512,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.use_mix = use_mix
        self.use_adv = use_adv
        self.mix_prob = mix_prob
        self.ot_type = ot_type
        self.ip_dim = ip_dim
        self.op_dim = op_dim

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        out = audio_output[1]["encoder_out"]
        audio_encoder_out = out["encoder_out"][0]
        audio_embedding = out["encoder_embedding"][0]
        audio_padding_mask = out["encoder_padding_mask"][0]
        loss, _, target = self.compute_loss(model, audio_output, sample, reduce=reduce)
        return loss, audio_output, target, audio_encoder_out.transpose(0, 1), audio_embedding, audio_padding_mask

    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        out = text_output[1]["encoder_out"]
        text_encoder_out = out["encoder_out"][0]
        text_embedding = out["encoder_embedding"][0]
        text_padding_mask = out["encoder_padding_mask"][0]
        loss, _, target = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss, text_output, target, text_encoder_out.transpose(0, 1), text_embedding, text_padding_mask

    def forward_decoder(self, model, sample, encoder_out, reduce):
        decoder_input = {
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "encoder_out": encoder_out,
        }
        decoder_output = model.decoder(**decoder_input)
        loss, _, target = self.compute_loss(model, decoder_output, sample, reduce=reduce)
        return loss, decoder_output, target

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        mix_loss, adv_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        st_size, mt_size, mix_size, adv_size = 0, 0, 0, 0

        if self.training:
            st_loss, st_output, st_target, st_encoder_out, st_emb, st_mask = self.forward_st(model, sample, reduce)
            mt_loss, mt_output, mt_target, mt_encoder_out, mt_emb, mt_mask = self.forward_mt(model, sample, reduce)
            st_lprobs, _, st_probs = self.get_lprobs_and_target(model, st_output, sample)
            mt_lprobs, _, mt_probs = self.get_lprobs_and_target(model, mt_output, sample)
            ot_st = self.get_ot_matrix(st_emb, st_mask, mt_emb, mt_mask)  # [B, T]
            mix_ot_out, mix_labels = self.get_mixed_sequence(st_encoder_out, mt_encoder_out, ot_st, self.mix_prob)  # [B, T, C]
            loss = st_loss + mt_loss
            if self.use_mix:
                mix_encoder_out = st_output[1]["encoder_out"]
                mix_encoder_out["encoder_out"][0] = mix_ot_out.transpose(0, 1)
                mix_loss, mix_output, mix_target = self.forward_decoder(model, sample, mix_encoder_out, reduce)
                mix_lprobs, _, mix_probs = self.get_lprobs_and_target(model, mix_output, sample)
                mix_loss = self.compute_mix_loss(st_lprobs, mt_lprobs, mix_lprobs, st_target, mt_target, mix_target)
                loss = loss + mix_loss
            if self.use_adv:
                adv_loss = self.compute_adv_loss(st_encoder_out, mt_encoder_out, mix_ot_out, mix_labels)
                loss = loss + adv_loss
            else:
                pass
            st_size = mt_size = mix_size = adv_size = sample_size = sample["ntokens"]
        else:
            st_loss, st_output, _, _, _, _ = self.forward_st(model, sample, reduce)
            loss = st_loss
            st_size = sample_size = sample["ntokens"]

        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "mt_loss": mt_loss.data,
            "mix_loss": mix_loss.data,
            "adv_loss": adv_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "st_sample_size": st_size,
            "mt_sample_size": mt_size,
            "mix_sample_size": mix_size,
            "adv_sample_size": adv_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, st_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = probs.log()
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), probs.view(-1, probs.size(-1))

    def distance(self, x, y, type):
        len1, len2 = x.size(-2), y.size(-2)
        bsz, dim = x.size(0), x.size(-1)
        tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
        ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)
        if type == "L2":
            dist = torch.linalg.norm(tx - ty, dim=-1)
            return dist
        else:
            sim = F.cosine_similarity(tx, ty, dim=-1)
            return 1. - sim

    def get_ot_matrix(self, x, x_padding_mask, y, y_padding_mask):
        # x: [B, T1, C], x_padding_mask: [B, T1]
        # y: [B, T2, C], y_padding_mask: [B, T2]
        dist = self.distance(x, y, type=self.ot_type)
        dist = dist.masked_fill(x_padding_mask.unsqueeze(-1), 6e4).masked_fill(y_padding_mask.unsqueeze(-2), 6e4)
        ot = dist.min(dim=-1)[1]
        return ot

    def get_mixed_sequence(self, x, y, ot, p=0.2):
        # x: [B, T1, C], y: [B, T2, C], ot: [B, T1]
        mixed = torch.zeros_like(x)
        mixlb = torch.zeros_like(x[:, :, 0], dtype=torch.float64)
        for i in range(x.size(0)):
            flag = random.random()
            for j in range(x.size(1)):
                if flag < 1 - p:
                    mixed[i, j, :] = x[i, j, :]
                    mixlb[i, j] = 1 - p
                else:
                    mixed[i, j, :] = y[i, ot[i, j], :]
                    mixlb[i, j] = p
        return mixed, mixlb.unsqueeze(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target, _ = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, target

    def compute_jsd_loss(self, st_lprobs, mt_lprobs, st_target, mt_target, ignore_index):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        st_pad_mask = st_target.eq(ignore_index)
        mt_pad_mask = mt_target.eq(ignore_index)
        kl_loss_st.masked_fill_(st_pad_mask, 0.0)
        kl_loss_mt.masked_fill_(mt_pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        jsd_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return jsd_loss

    def compute_mix_loss(self, st_lprobs, mt_lprobs, mix_lprobs, st_target, mt_target, mix_target):
        jsd_s_loss = self.compute_jsd_loss(st_lprobs, mix_lprobs, st_target, mix_target, self.padding_idx)
        jsd_t_loss = self.compute_jsd_loss(mt_lprobs, mix_lprobs, mt_target, mix_target, self.padding_idx)
        mix_loss = (jsd_s_loss + jsd_t_loss) / 2.0
        return mix_loss

    def compute_adv_loss(self, st_out, mt_out, mix_out, mix_labels):
        adv_model = Discriminator(self.ip_dim, self.op_dim).to(mix_out.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum").to(mix_out.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            adv_mix_out, adv_st_out, adv_mt_out = adv_model(mix_out), adv_model(st_out), adv_model(mt_out)
            mix_adv_loss = loss_fn(adv_mix_out, mix_labels)
            st_adv_loss = loss_fn(adv_st_out, torch.full_like(adv_st_out, 0.5))
            mt_adv_loss = loss_fn(adv_mt_out, torch.full_like(adv_mt_out, 0.5))
            adv_loss = mix_adv_loss + st_adv_loss + mt_adv_loss
        return adv_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target, _ = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        mix_loss_sum = sum(log.get("mix_loss", 0) for log in logging_outputs)
        adv_loss_sum = sum(log.get("adv_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        mix_sample_size = sum(log.get("mix_sample_size", 0) for log in logging_outputs)
        adv_sample_size = sum(log.get("adv_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "mix_loss", mix_loss_sum / mix_sample_size / math.log(2) if mix_sample_size != 0 else 0, mix_sample_size, round=3
        )
        metrics.log_scalar(
            "adv_loss", adv_loss_sum / adv_sample_size / math.log(2) if adv_sample_size != 0 else 0, adv_sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
