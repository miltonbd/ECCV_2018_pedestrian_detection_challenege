from __future__ import print_function

from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os
import torch
from utils.utils import progress_bar
from torchvision import transforms
from dataloader import Normalizer, Resizer
import  skimage
def evaluate_wider_pedestrian_for_upload(epoch, dataset, model_new,retinanet_sk, threshold=0.5):
    print("\n==> Testing wider pedestrian dataset.")
    new_model_1 = model_new._modules['module']
    state_dict_new = new_model_1.state_dict()
    retinanet_sk.load_state_dict(state_dict_new)
    model1=copy.deepcopy(retinanet_sk).cuda(0)

    model1.eval()


    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        scores_for_upload = []
        for index, data in enumerate(dataset):
            img = skimage.io.imread(dataset[index])
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)

            img=img.astype(np.float32) / 255.0
            data_fake={'img':img,'annot':np.random.rand(2,4)}
            # scale = data['scale']
            progress_bar(index, len(dataset), "Evaluating........")
            transform = transforms.Compose([Normalizer(), Resizer()])
            data_resized=transform(data_fake)
            scale = data_resized['scale']

            # run network
            scores, labels, boxes = model1(data_resized['img'].permute(2,0,1).cuda(0).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            # change to (x, y, w, h) (MS COCO standard)
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

            # compute predicted labels and scores
            # for box, score, label in zip(boxes[0], scores[0], labels[0]):
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                box_list = box.tolist()
                # image_result = {
                #     'image_id': dataset.image_ids[index],
                #     'category_id': dataset.label_to_coco_label(label),
                #     'score': float(score),
                #     'bbox': box_list,
                # }
                image_name = os.path.basename(dataset[index])
                score_row = "{img} {score:.3f} {xmin:.1f} {ymin:.1f} {w:.1f} {h:.1f}".format(
                    img=image_name, score=score, xmin=box_list[0], ymin=box_list[1], w=box_list[2], h=box_list[3])
                scores_for_upload.append(score_row)
                # append detection to results
                # results.append(image_result)

            # append image to list of processed images
        from utils.file_utils import save_to_file
        submit_file = 'res/scores.txt'
        save_to_file(submit_file, scores_for_upload)
        # import shutil
        # shutil.make_archive('res/scores.txt', 'zip', 'res')

        import zipfile
        archive = zipfile.ZipFile("res/scores.txt.zip", "w")
        try:
            archive.write(submit_file)
            print('Files added.')
        finally:
            print('Reading files now.')
            archive.close()
        model1.train()
        return

import copy
def evaluate_wider_pedestrian(epoch, dataset, model_new,retinanet_sk, threshold=0.5):
    print("\n==> Evaluating wider pedestrian dataset.")
    new_model_1 = model_new._modules['module']
    state_dict_new = new_model_1.state_dict()
    retinanet_sk.load_state_dict(state_dict_new)
    model1=copy.deepcopy(retinanet_sk).cuda(0)

    model1.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        scores_for_upload=[]
        for index in range(len(dataset)):
            # if index>50:
            #     break
            data = dataset[index]
            scale = data['scale']
            progress_bar(index, len(dataset), "Evaluating........")

            # run network
            data1 = data['img'].permute(2, 0, 1).cuda(0)
            scores, labels, boxes = model1(data1.float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            # change to (x, y, w, h) (MS COCO standard)
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

            # compute predicted labels and scores
            # for box, score, label in zip(boxes[0], scores[0], labels[0]):
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                box_list = box.tolist()
                image_result = {
                    'image_id': dataset.image_ids[index],
                    'category_id': dataset.label_to_coco_label(label),
                    'score': float(score),
                    'bbox': box_list,
                }
                image_name=os.path.basename(dataset.get_image_name(index))
                score_row="{img} {score:.3f} {xmin:.1f} {ymin:.1f} {w:.1f} {h:.1f}".format(
                    img=image_name,score=score,xmin=box_list[0],ymin=box_list[1],w=box_list[2],h=box_list[3])
                scores_for_upload.append(score_row)
                # append detection to results
                results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])
        from utils.file_utils import save_to_file
        save_to_file('submit_files/scores_validation.txt',scores_for_upload)
        # print progress
        if not len(results):
            return


        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        summary=coco_eval.summarize()
        validation_score=-1
        from score_pedestrian_detection import get_average_precision_validation

        return get_average_precision_validation()

def evaluate_coco(dataset, model, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            progress_bar(index,len(dataset),"Evaluating........")

            # run network
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            # change to (x, y, w, h) (MS COCO standard)
            if boxes.shape[0]>0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

            # compute predicted labels and scores
            #for box, score, label in zip(boxes[0], scores[0], labels[0]):
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                image_result = {
                    'image_id'    : dataset.image_ids[index],
                    'category_id' : dataset.label_to_coco_label(label),
                    'score'       : float(score),
                    'bbox'        : box.tolist(),
                }

                # append detection to results
                results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return
