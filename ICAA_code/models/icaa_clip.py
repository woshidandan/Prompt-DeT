import torch
import torch.nn as nn

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from .clip import clip
from .curve_mapping import CurveMapping


def load_clip_to_cpu(model_path="checkpoint/RN50.pt", pretrained=True):
    # loading JIT archive
    jit_model = torch.jit.load(model_path, map_location="cpu")
    state_dict = jit_model.state_dict()
    model = clip.build_model(state_dict)
    if pretrained is True:
        model.load_state_dict(state_dict)

    return model


class ICAA_CLIP(nn.Module):
    def __init__(
        self,
        pos_embedding=False,
        classnames=[
            ["beautiful.", "ugly."],
            ["colorful.", "monotone."],
            ["warm tone.", "cold tone."],
            ["harmonious color.", "chaotic color."],
        ],
        means=[
            0.5430292152312776,
            0.5746566870547009,
            0.5060040414776488,
            0.5041112922830655,
        ],
        stds=[
            0.026028287910949126,
            0.04772679232711666,
            0.021420622256879244,
            0.024548609685370107,
        ],
        pretrained_clip_path="checkpoint/RN50.pt",
        clip_pretrained=True,
        adapter_finetune=False,
        curve_mapping=False,
    ):
        super().__init__()

        self.clip_model = load_clip_to_cpu(model_path=pretrained_clip_path, pretrained=clip_pretrained)
        self.clip_model.requires_grad_(False)
        self.pos_embedding = pos_embedding
        self.tokenized_prompts = []
        self.num_clip = len(classnames)
        for i in range(self.num_clip):
            self.tokenized_prompts.append(clip.tokenize(classnames[i]))
        self.adapter_finetune = adapter_finetune
        self.curve_mapping = curve_mapping
        if self.adapter_finetune:
            self.alpha = nn.Parameter(torch.Tensor([0.5]))
            self.beta = nn.Parameter(torch.Tensor([0.5]))
            self.image_adapter = nn.Sequential()
            self.image_adapter.add_module("linear1", nn.Linear(1024, 4096))
            self.image_adapter.add_module("act", nn.GELU())
            self.image_adapter.add_module("linear2", nn.Linear(4096, 1024))
            self.text_adapter = nn.Sequential()
            self.text_adapter.add_module("linear1", nn.Linear(1024, 4096))
            self.text_adapter.add_module("act", nn.GELU())
            self.text_adapter.add_module("linear2", nn.Linear(4096, 1024))
        if self.curve_mapping:
            self.curve_mappings = nn.ModuleList()
            for i in range(self.num_clip):
                self.curve_mappings.append(CurveMapping())
        else:
            self.means = means
            self.stds = stds
            self.mean = 1.0 / 2.0
            self.std = 1.0 / 6.0

    def distrib_mapping(self, x, idx):
        return (x - self.means[idx]) * (self.std / self.stds[idx]) + self.mean
        # return x

    def encode_image(self, image, pos_embedding):
        return self.clip_model.encode_image(image, pos_embedding)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def clip_forward(self, image, text, pos_embedding=False):
        image_features = self.encode_image(image, pos_embedding)
        text_features = self.encode_text(text)
        if self.adapter_finetune:
            image_features = self.image_adapter(image_features) * self.alpha + image_features * (1 - self.alpha)
            text_features = self.text_adapter(text_features) * self.beta + text_features * (1 - self.beta)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def forward(self, x):
        if self.clip_model.training:
            self.clip_model.eval()

        logits_list = []
        for i in range(self.num_clip):
            logits_per_image, _ = self.clip_forward(x, self.tokenized_prompts[i].to(x.device), self.pos_embedding)
            probs = logits_per_image / torch.sum(logits_per_image, dim=-1, keepdim=True)
            if self.curve_mapping:
                probs = self.curve_mappings[i](probs)
            else:
                probs = self.distrib_mapping(probs, i)
            probs = torch.clip(probs, 0, 1)
            logits_list.append(probs[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=1).float()

        return logits_list
