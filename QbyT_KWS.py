from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import jiwer
import torch
import string
from tqdm import trange
import numpy as np


def process_audio(tc, processor, model, device):
    sample = tc["test_audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features.to(device)

    predicted_ids = model.generate(input_features)
    hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    hypothesis = hypothesis.strip().lower().translate(str.maketrans("", "", ',.-?!:"'))
    return hypothesis


def min_cer(ref, hyp):
    ref_l = len(ref.split(" "))
    hyp_split = hyp.split(" " )
    if len(hyp_split) <= ref_l:
        return jiwer.cer(ref, hyp)

    ret = float("inf")
    for i in range(len(hyp_split)-ref_l+1):
        ret = min(ret, jiwer.cer(ref, " ".join(hyp_split[i:i+ref_l])))

    return ret


def get_accuracy(cer, label):
    return (label == 0 and cer > 0.0) or (label == 1 and cer == 0.0)


def get_fnr_fpr(results, thresholds):
    pos = max(np.count_nonzero(results[:, 1]==1), 1)
    neg = max(np.count_nonzero(results[:, 1]==0), 1)

    fnr = np.array([np.count_nonzero((results[:, 1]==1) & (results[:, 0]>t))/pos for t in thresholds])
    fpr = np.array([np.count_nonzero((results[:, 1]==0) & (results[:, 0]<=t))/neg for t in thresholds])

    return fnr, fpr


def get_eer(cers, labels):
    thresholds = np.arange(0., 1., 0.01)
    fnr, fpr = get_fnr_fpr(np.array(list(zip(cers, labels))), thresholds)
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))

    return 100*np.max([fnr[eer_idx], fpr[eer_idx]])


def process_ds(ds, processor, model, device):
    with trange(len(ds)) as t:
        tot_accuracy = 0
        cers = []
        labels = []
        eer = 100.
        for i in t:
            hypothesis = process_audio(ds[i], processor, model, device)
            cer = min_cer(ds[i]["keyword_transcription"], hypothesis)
            tot_accuracy += get_accuracy(cer, ds[i]["label"])
            cers.append(cer)
            labels.append(ds[i]["label"])
            t.set_description("Accuracy so far: %.2f, EER so far: %.2f"%(100*tot_accuracy/(i+1), eer))

            if i%10 == 0:
                eer = get_eer(cers, labels)

        print("Final EER: %.2f"%(eer))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model = model.to(device)

    ds = load_dataset("voiceintelligenceresearch/MOCKS", "en.LS-other.subset", split="offline", revision="67ea4e6b")
    process_ds(ds, processor, model, device)

    ds = load_dataset("voiceintelligenceresearch/MOCKS", "en.LS-clean.subset", split="offline", revision="67ea4e6b")
    process_ds(ds, processor, model, device)

    ds = load_dataset("voiceintelligenceresearch/MOCKS", "en.MCV.subset", split="offline", revision="67ea4e6b")
    process_ds(ds, processor, model, device)

