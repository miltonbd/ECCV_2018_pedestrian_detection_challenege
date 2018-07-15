import numpy as np
import os
import argparse
import os.path as osp

def check_size(submission_file):
    max_size = 60*1024*1024
    if osp.getsize(submission_file) > max_size:
        raise IOError #File size exceeds the specified maximum size, which is 60M for the server.

def judge_overlap(pbox,ignore_box):
    overlap=[]
    delete=[]
    for p in pbox:
        pl=min(p[0],p[2])
        pr=max(p[0],p[2])
        pb=min(p[1],p[3])
        pt=max(p[1],p[3])
        s_p=(pr-pl)*(pt-pb)
        s_lap=-0.01
        for c in ignore_box:
            cl=min(c[0],c[2])
            cr=max(c[0],c[2])
            cb=min(c[1],c[3])
            ct=max(c[1],c[3])
            if not (cr<pl or cl>pr or ct<pb or cb>pt):
                s_lap+=(min(cr,pr)-max(cl,pl))*(min(ct,pt)-max(cb,pb))
        if s_lap>0:
            overlap.append([p,s_lap/s_p])
    for o in overlap:
        if o[1]>0.5:
            delete.append(o[0])
    remain_id = [p for p in pbox if p not in delete]
    return remain_id

def parse_ignore_file(ignore_file):
    with open(ignore_file,'r') as f:
        lines = f.readlines()
    ig = [x.strip().split() for x in lines]
    ignore = {}
    for item in ig:
        key = item[0]
        ignore_num = (len(item)-1)/4
        bbox = []
        for i in range(int(ignore_num)):
            b = []
            b.append(int(item[1+4*i]))
            b.append(int(item[2+4*i]))
            b.append(int(item[1+4*i])+int(item[3+4*i]))
            b.append(int(item[2+4*i])+int(item[4+4*i]))
            bbox.append(b)
        ignore[key] = bbox
    return ignore

def parse_submission(submission_file,ignore_file):
    ignore_zone = parse_ignore_file(ignore_file)
    ignore_keys = ignore_zone.keys()
    with open(submission_file, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split() for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = []
    for x in splitlines:
        bb = []
        bb.append(float(x[2]))
        bb.append(float(x[3]))
        bb.append(float(x[2])+float(x[4]))
        bb.append(float(x[3])+float(x[5]))
        BB.append(bb)

    sub_key = []
    for x in image_ids:
        if x not in sub_key:
            sub_key.append(x)
    final_confidence = []
    final_ids = []
    final_BB = []

    for key in sub_key:
        find = [i for i,v in enumerate(image_ids) if v == key]
        BB_sub = [BB[i] for i in find]
        confid_sub = [confidence[i] for i in find]
        if key in ignore_keys:
            ignore_bbox = ignore_zone[key]
            bbox_remain = judge_overlap(BB_sub,ignore_bbox)
            find_remain = []
            for i,v in enumerate(BB_sub):
                if v in bbox_remain:
                    find_remain.append(i)
            confid_remain = [confid_sub[i] for i in find_remain]
            BB_sub = bbox_remain
            confid_sub = confid_remain
        ids_sub = [key]*len(BB_sub)
        final_ids.extend(ids_sub)
        final_confidence.extend(confid_sub)
        final_BB.extend(BB_sub)

    final_BB = np.array(final_BB)
    final_confidence = np.array(final_confidence)
    sorted_ind = np.argsort(-final_confidence)
    final_BB = final_BB[sorted_ind, :]
    final_ids = [final_ids[x] for x in sorted_ind]
    return final_ids, final_BB

def parse_gt_annotation(gt_file,ignore_file):
    ignore_zone = parse_ignore_file(ignore_file)
    ignore_keys = ignore_zone.keys()
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    info = [x.strip().split() for x in lines]
    gt = {}
    for item in info:
        bbox = []
        bbox_num = (len(item)-1)/5
        for i in range(int(bbox_num)):
            b = []
            b.append(int(item[2+5*i]))
            b.append(int(item[3+5*i]))
            b.append(int(item[2+5*i])+int(item[4+5*i]))
            b.append(int(item[3+5*i])+int(item[5+5*i]))
            bbox.append(b)
        if item[0] in ignore_keys:
            ignore_bbox = ignore_zone[item[0]]
            bbox_remain = judge_overlap(bbox,ignore_bbox)
        else:
            bbox_remain = bbox
        gt[item[0]] = np.array(bbox_remain)
    return gt

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def pedestrian_eval(input, gt_file, ignore_file, ovthresh):
    gt = parse_gt_annotation(gt_file,ignore_file)
    image_ids, BB = parse_submission(input,ignore_file)
    npos = 0
    recs = {}
    for key in gt.keys():
        det = [False]*len(gt[key])
        recs[key] = {'bbox': gt[key], 'det': det}
        npos += len(gt[key])
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in recs.keys():
            raise KeyError("Can not find image {} in the groundtruth file, did you submit the result file for the right dataset?".format(image_ids[d]))
    for d in range(nd):
        R = recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos+1e-8)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec)
    return ap


def wider_ped_eval(input, gt,ignore_file):
    aap = []
    for ove in np.arange(0.5, 1.0, 0.05):
        ap = pedestrian_eval(input, gt,ignore_file, ovthresh=ove)
        aap.append(ap)
    mAP = np.average(aap)
    return mAP


def get_average_precision_validation():
    input_dir = './'
    output_dir = './'
    ref_dir = osp.join(input_dir, 'ref')
    submit_dir = osp.join(input_dir, 'res')
    submit_file = 'submit_files/scores_validation.txt'
    gt_file = osp.join(ref_dir, 'val_annotations.txt')
    ignore_file = osp.join(ref_dir, 'pedestrian_ignore_part_val.txt')
    check_size(submit_file)
    mAP = wider_ped_eval(submit_file, gt_file, ignore_file)
    out = {'Average AP': mAP}
    print(out)
    return mAP


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input", type=str)
    # parser.add_argument("output", type=str)
    # args = parser.parse_args()
    get_average_precision_validation()
    # strings = ['{}: {}\n'.format(k, v) for k, v in out.items()]
    # open(os.path.join(output_dir, 'scores.txt'), 'w').writelines(strings)