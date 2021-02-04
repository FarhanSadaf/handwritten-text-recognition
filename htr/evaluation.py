import stringdist
import numpy as np
import string


def ocr_metrics(pred_texts, gt_texts, lower=True):
    '''
    lower: If set, converted to lowercase
    Takes 'predicted-texts' and 'ground truth-texts' as arguments.
    Returns 
        Character Error Rate (CER)
        Word Error Rate (WER)
        Sequence Error Rate (SER)
    '''
    cer, wer, ser = [], [], []
    for pred, gt in zip(pred_texts, gt_texts):
        if lower:
            pred, gt = pred.lower(), gt.lower()
            
        # CER
        pred_cer, gt_cer = list(pred), list(gt)
        dist = stringdist.levenshtein(pred_cer, gt_cer)
        cer.append(dist / max(len(pred_cer), len(gt_cer)))

        # WER
        pred_wer, gt_wer = pred.split(), gt.split()
        dist = stringdist.levenshtein(pred_wer, gt_wer)
        wer.append(dist / max(len(pred_wer), len(gt_wer)))

        # SER
        pred_ser, gt_ser = [pred], [gt]
        dist = stringdist.levenshtein(pred_ser, gt_ser)
        ser.append(dist / max(len(pred_ser), len(gt_ser)))

    return np.mean([cer, wer, ser], axis=1)
