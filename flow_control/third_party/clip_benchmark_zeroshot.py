"""Vendored zero-shot classification helpers from clip-benchmark.

Source:
    https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_classification.py
    (itself adapted from open_clip's ``src/training/zero_shot.py``).

Only ``zero_shot_classifier`` and ``run_classification`` are vendored here -- the
two functions GenEval's colour classifier relies on. The upstream module also
exports ``accuracy`` / ``average_precision_per_class`` / ``evaluate``; ``evaluate``
is the sole reason ``clip-benchmark`` depends on ``scikit-learn`` (it calls
``balanced_accuracy_score`` / ``classification_report``). Dropping it lets us avoid
that dependency -- and its ``scikit-learn<2`` upper cap, plus ``pycocoevalcap`` and
``webdataset`` -- entirely.

The upstream ``tqdm`` progress bars have been dropped: GenEval is the only caller
and ran with them disabled anyway.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.


    model:
        CLIP-like model with `encode_text`

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    classnames: list of str
        name of classes

    templates: list of str
        templates to use.

    Returns
    -------

    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    with torch.no_grad(), torch.autocast(device, enabled=amp):
        zeroshot_weights = []
        for classname in classnames:
            if isinstance(templates, dict):
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif isinstance(templates, list):
                # generic prompts that are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`

    dataloader: torch.utils.data.Dataloader

    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    pred = []
    true = []
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device)

            with torch.autocast(device, enabled=amp):
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ classifier

            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true
