# from __future__ import print_function, division
# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from pycocotools.coco import COCO
# from typing import List
# import math
# from PIL import Image

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# class MyDataMaskRCNNDataset(Dataset):
#     """
#         __init__ : Initialize the dataset
#         1. Get the image paths
#         2. Get the annotation paths
#         3. Get the transform
#         4. Get the image height and width
#         5. Generate indexes for the image patches (image_index, (patch_index_x, patch_index_y))

#         __getitem__ : Get the image and the target
#         1. Get the image path and the annotation path with image_index
#         2. Read the image and the annotation
#         3. Pad the image and the annotation to the image_height and image_width
#         4. Get the bounding boxes and the labels
#         5. Get the patch from the image, modify the bounding box according to the patch
#         6. Get the target mask batch
#         7. Resize the image to a LR image
#         8. Convert the image and the target to tensor
#         9. Return the image and the target
#     """
#     def __init__(
#           self, 
#           data_dir: str,
#           meta_data_dir: str,
#           mask_dir: str,
#           image_height: int=6400, 
#           image_width: int=6400, 
#           patch_height: int=800, 
#           patch_width: int=800, 
#           transform = None) -> None:
#         super().__init__()

#         self.data_dir = data_dir
#         self.annotation_dir = meta_data_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.image_height = image_height
#         self.image_width = image_width
#         self.patch_height = patch_height
#         self.patch_width = patch_width

#         self.coco_data = COCO(self.annotation_dir)
        
#         self.total_classes = len(self.coco_data.getCatIds())

#         self._get_index_list(self.coco_data.getImgIds())

#     def __len__(self) -> np.int:
#         return len(self.index_list)

#     def _get_index_list(self, img_idxs: List[int]):
#         self.index_list = []
#         total_height_patches = self.image_height // self.patch_height
#         total_width_patches = self.image_width // self.patch_width
#         for i in img_idxs:
#             for j in range(total_height_patches):
#                 for k in range(total_width_patches):
#                     self.index_list.append((i, (j, k)))

#     def _change_bbox_xyhw_to_xyxy(self, annotations):
#         for annotation in annotations:
#             x, y, w, h = annotation['bbox']
#             annotation['bbox'] = [x, y, x + w, y + h]
#         return annotations

#     def _change_bbox_xyxy_to_xyhw(self, annotations):
#         for annotation in annotations:
#             x_min, y_min, x_max, y_max = annotation['bbox']
#             annotation['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
#         return annotations
    
#     def _get_mask(self, mask_path):
#         mask_sparse = torch.load(mask_path)
#         mask_sparse.update({"values": torch.ones(mask_sparse["nnz"])})
#         del mask_sparse["nnz"]
#         mask = torch.sparse_coo_tensor(**mask_sparse)
#         mask = mask.to_dense()
#         mask = mask.unsqueeze(1)
#         return mask.numpy()

#     def _generate_mask(self, annotations, patch_bbox, mask_path):
#         mask = []
#         image = []
#         overlapping_annotations = []
#         mask = self._get_mask(mask_path)
#         for i, annotation in enumerate(annotations):
#             temp = mask[i].squeeze()
#             zero_temp = np.zeros((6400, 6400))
#             zero_temp[200:6200, 200:6200] = temp
#             temp = zero_temp
#             if not np.sum(temp[patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]]) == 0:
#                 image.append(np.expand_dims(temp, axis=0))
#                 overlapping_annotations.append(annotation)
#         rectified_annotations = []
#         rectified_masks = []
#         for ann, img in zip(overlapping_annotations, image):
#             # print(ann['bbox'])
#             ann['bbox'] = [ann['bbox'][0] - patch_bbox[0], ann['bbox'][1] - patch_bbox[1],
#                            ann['bbox'][2] , ann['bbox'][3]]
#             x1, y1, x2, y2 = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]
#             temp_box = [x1, y1, x2, y2]
#             temp_box = np.clip(temp_box, 0, 800)
#             x = temp_box[0]
#             y = temp_box[1]
#             w = temp_box[2] - temp_box[0]
#             h = temp_box[3] - temp_box[1]
#             ann['bbox'] = [x, y, w, h]
#             if w*h > 0:
#                 rectified_annotations.append(ann)
#                 rectified_masks.append(img)
#         try:
#             mask = np.stack(rectified_masks, axis=0)
#             mask = mask[:, :, patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]]
#         except:
#             mask = np.zeros((1, 1, 800, 800))
#         # Show the mask
#         # plt.figure(figsize=(10, 10))
#         # t = mask[:, 0, :,:]
#         # t = np.sum(t, axis=0)
#         # plt.imshow(t[patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]])
#         # print(overlapping_annotations)
#         return rectified_annotations, mask

#     def __getitem__(self, idx):
#         image_index, patch_index = self.index_list[idx]
#         image_data = self.coco_data.loadImgs(image_index)[0]
#         annotation_ids = self.coco_data.getAnnIds(imgIds=image_index)
#         annotations = self.coco_data.loadAnns(annotation_ids)
#         image_path = os.path.join(self.data_dir, image_data['file_name'])
#         mask_path = os.path.join(self.mask_dir, image_data['file_name'].split('.')[0] + '.pt')
#         # print(image_path)
#         # image = cv2.imread(image_path, 1)
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = Image.open(image_path).convert('RGB')
#         image  = np.array(image)
#         image = np.pad(image, ((200, 200), (200, 200), (0, 0)), 
#                        mode='constant', constant_values=0)
        
#         ## NOTE: First two coordinates are x, y, need to add 200 to each
#         ## NOTE: All annotation coordinates need to be updated (add 200 to all)
#         for annotation in annotations:
#             annotation['bbox'][0] += 200
#             annotation['bbox'][1] += 200
#         patch_x, patch_y = patch_index[0] * self.patch_height, patch_index[1] * self.patch_width
#         patch_bbox = [patch_y, patch_x, patch_y + self.patch_height, patch_x + self.patch_width]
#         overlapping_annotations, mask = self._generate_mask(annotations, patch_bbox, mask_path)
#         patch_img = image[patch_x:patch_x + self.patch_height, patch_y:patch_y + self.patch_width, :]
#         overlapping_annotations = self._change_bbox_xyhw_to_xyxy(overlapping_annotations)
#         patch_img_lr = self.imresize_np(patch_img, 0.25)

#         boxes = []
#         labels = []
#         area = []
#         iscrowd = []
#         for annotation in overlapping_annotations:
#             boxes.append(annotation['bbox'])
#             labels.append(annotation['category_id'])
#             area.append(annotation['area'])
#             iscrowd.append(annotation['iscrowd'])

#         if len(boxes) == 0:
#             boxes = [[0, 0, 1, 1]]
#             labels = [0]
#             area = [1]
#             iscrowd = [0]
#         target = {}
#         target['image'] = patch_img
#         target['masks'] = list(mask) if mask is not None else []
#         target['image_lq'] = patch_img_lr
#         target['bboxes'] = boxes
#         target['labels'] = labels
#         target['area'] = area
#         target['iscrowd'] = iscrowd
#         target['image_id'] = idx
#         name = image_data['file_name'].split('.')[0]
#         name = name + '_patch_' + str(patch_index[0]) + '_' + str(patch_index[1]) + '.jpg'
#         lq_path = os.path.join(self.data_dir, name)
#         target['LQ_path'] = lq_path

#         if self.transform is None:
#             image, target = self.convert_to_tensor(**target)
#             return image, target
#         else:
#             for i in range(len(target['masks'])):
#                 target['masks'][i] = target['masks'][i][0]
#             transformed = self.transform(**target)
#             for i in range(len(transformed['masks'])):
#                 transformed['masks'][i] = np.expand_dims(transformed['masks'][i], axis=0)
#             image, target = self.convert_to_tensor(**transformed)
#             return image, target

#     def cubic(self, x):
#         absx = torch.abs(x)
#         absx2 = absx**2
#         absx3 = absx**3
#         return (1.5 * absx3 - 2.5 * absx2 + 1) * (
#             (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
#                 (absx > 1) * (absx <= 2)).type_as(absx))
    
#     def calculate_weights_indices(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
#         if (scale < 1) and (antialiasing):
#             # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
#             kernel_width = kernel_width / scale

#         # Output-space coordinates
#         x = torch.linspace(1, out_length, out_length)

#         # Input-space coordinates. Calculate the inverse mapping such that 0.5
#         # in output space maps to 0.5 in input space, and 0.5+scale in output
#         # space maps to 1.5 in input space.
#         u = x / scale + 0.5 * (1 - 1 / scale)

#         # What is the left-most pixel that can be involved in the computation?
#         left = torch.floor(u - kernel_width / 2)

#         # What is the maximum number of pixels that can be involved in the
#         # computation?  Note: it's OK to use an extra pixel here; if the
#         # corresponding weights are all zero, it will be eliminated at the end
#         # of this function.
#         P = math.ceil(kernel_width) + 2

#         # The indices of the input pixels involved in computing the k-th output
#         # pixel are in row k of the indices matrix.
#         indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
#             1, P).expand(out_length, P)

#         # The weights used to compute the k-th output pixel are in row k of the
#         # weights matrix.
#         distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
#         # apply cubic kernel
#         if (scale < 1) and (antialiasing):
#             weights = scale * self.cubic(distance_to_center * scale)
#         else:
#             weights = self.cubic(distance_to_center)
#         # Normalize the weights matrix so that each row sums to 1.
#         weights_sum = torch.sum(weights, 1).view(out_length, 1)
#         weights = weights / weights_sum.expand(out_length, P)

#         # If a column in weights is all zero, get rid of it. only consider the first and last column.
#         weights_zero_tmp = torch.sum((weights == 0), 0)
#         if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
#             indices = indices.narrow(1, 1, P - 2)
#             weights = weights.narrow(1, 1, P - 2)
#         if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
#             indices = indices.narrow(1, 0, P - 2)
#             weights = weights.narrow(1, 0, P - 2)
#         weights = weights.contiguous()
#         indices = indices.contiguous()
#         sym_len_s = -indices.min() + 1
#         sym_len_e = indices.max() - in_length
#         indices = indices + sym_len_s - 1
#         return weights, indices, int(sym_len_s), int(sym_len_e)

#     def imresize_np(self, img, scale, antialiasing=True):
#         # Now the scale should be the same for H and W
#         # input: img: Numpy, HWC BGR [0,1]
#         # output: HWC BGR [0,1] w/o round
#         img = torch.from_numpy(img)

#         in_H, in_W, in_C = img.size()
#         _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
#         kernel_width = 4
#         kernel = 'cubic'

#         # Return the desired dimension order for performing the resize.  The
#         # strategy is to perform the resize first along the dimension with the
#         # smallest scale factor.
#         # Now we do not support this.

#         # get weights and indices
#         weights_H, indices_H, sym_len_Hs, sym_len_He = self.calculate_weights_indices(
#             in_H, out_H, scale, kernel, kernel_width, antialiasing)
#         weights_W, indices_W, sym_len_Ws, sym_len_We = self.calculate_weights_indices(
#             in_W, out_W, scale, kernel, kernel_width, antialiasing)
#         # process H dimension
#         # symmetric copying
#         img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
#         img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

#         sym_patch = img[:sym_len_Hs, :, :]
#         inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(0, inv_idx)
#         img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

#         sym_patch = img[-sym_len_He:, :, :]
#         inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(0, inv_idx)
#         img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

#         out_1 = torch.FloatTensor(out_H, in_W, in_C)
#         kernel_width = weights_H.size(1)
#         for i in range(out_H):
#             idx = int(indices_H[i][0])
#             out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
#             out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
#             out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

#         # process W dimension
#         # symmetric copying
#         out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
#         out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

#         sym_patch = out_1[:, :sym_len_Ws, :]
#         inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(1, inv_idx)
#         out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

#         sym_patch = out_1[:, -sym_len_We:, :]
#         inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(1, inv_idx)
#         out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

#         out_2 = torch.FloatTensor(out_H, out_W, in_C)
#         kernel_width = weights_W.size(1)
#         for i in range(out_W):
#             idx = int(indices_W[i][0])
#             out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
#             out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
#             out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

#         return out_2.numpy()

#     def convert_to_tensor(self, **target):
#         #convert to tensor
#         target['image_lq'] = torch.from_numpy(target['image_lq'].transpose((2, 0, 1)))
#         target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
#         target['boxes'] = torch.tensor(target['bboxes'], dtype=torch.float32)
#         target['masks'] = torch.tensor(target['masks'], dtype=torch.float32)
#         target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
#         target["area"] = torch.tensor(target['area'])
#         target["iscrowd"] = torch.tensor(target['iscrowd'])
#         target['image_id'] = torch.tensor([target['image_id']])


#         image = {}
#         image['image_lq'] = target['image_lq']
#         image['image'] = target['image']
#         image['LQ_path'] = target['LQ_path']

#         del target['image_lq']
#         del target['image']
#         del target['bboxes']
#         del target['LQ_path']

#         return image, target



from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from typing import List
import math
from PIL import Image
import sys

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MyDataMaskRCNNDataset(Dataset):
    """
        __init__ : Initialize the dataset
        1. Get the image paths
        2. Get the annotation paths
        3. Get the transform
        4. Get the image height and width
        5. Generate indexes for the image patches (image_index, (patch_index_x, patch_index_y))

        __getitem__ : Get the image and the target
        1. Get the image path and the annotation path with image_index
        2. Read the image and the annotation
        3. Pad the image and the annotation to the image_height and image_width
        4. Get the bounding boxes and the labels
        5. Get the patch from the image, modify the bounding box according to the patch
        6. Get the target mask batch
        7. Resize the image to a LR image
        8. Convert the image and the target to tensor
        9. Return the image and the target
    """
    def __init__(
          self, 
          data_dir: str,
          meta_data_dir: str,
          mask_dir: str,
          image_height: int=500, 
          image_width: int=500, 
          transform = None,
          verbose = "train") -> None:
        super().__init__()

        self.data_dir = data_dir
        self.annotation_dir = meta_data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.verbose = verbose

        self.coco_data = COCO(self.annotation_dir)
        
        self.total_classes = len(self.coco_data.getCatIds())

        self.index_list = []
        for idx in self.coco_data.getImgIds():
            if len(self.coco_data.getAnnIds(imgIds=idx)) == 0:
                continue
            # elif idx == 11938:
            #     print(idx, "**********")
            #     continue
            else:
                self.index_list.append(idx)


        self.index_list = self.index_list#[:20]
        # with open("index_list.txt", "w") as f:
        #     for idx in self.index_list:
        #         img = self.coco_data.loadImgs([idx])[0]
        #         ann = self.coco_data.getAnnIds(imgIds=[idx])
        #         print(f"{img['file_name']}, {img['id']}, {ann}")
        #         f.write(f"{img['file_name']}, {img['id']}\n")
        # f.close()

        # sys.exit(0)

    def __len__(self) -> np.int:
        return len(self.index_list)

    # def _get_index_list(self, img_idxs: List[int]):
    #     self.index_list = []
    #     total_height_patches = self.image_height // self.patch_height
    #     total_width_patches = self.image_width // self.patch_width
    #     for i in img_idxs:
    #         for j in range(total_height_patches):
    #             for k in range(total_width_patches):
    #                 self.index_list.append((i, (j, k)))

    def _change_bbox_xyhw_to_xyxy(self, annotations):
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            annotation['bbox'] = [x, y, x + w, y + h]
            # print(self.verbose, "********", temp)
        return annotations

    # def _change_bbox_xyxy_to_xyhw(self, annotations):
    #     for annotation in annotations:
    #         x_min, y_min, x_max, y_max = annotation['bbox']
    #         annotation['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
    #     return annotations
    
    def _get_mask(self, mask_path):
        mask_sparse = torch.load(mask_path)
        if mask_sparse["nnz"] == 0:
            return np.zeros((1, 1, 500, 500))
        mask_sparse.update({"values": torch.ones(mask_sparse["nnz"])})
        del mask_sparse["nnz"]
        mask = torch.sparse_coo_tensor(**mask_sparse)
        mask = mask.to_dense()
        return mask.numpy()

    # def _generate_mask(self, annotations, patch_bbox, mask_path):
    #     mask = []
    #     image = []
    #     overlapping_annotations = []
    #     mask = self._get_mask(mask_path)
    #     for i, annotation in enumerate(annotations):
    #         temp = mask[i].squeeze()
    #         zero_temp = np.zeros((6400, 6400))
    #         zero_temp[200:6200, 200:6200] = temp
    #         temp = zero_temp
    #         if not np.sum(temp[patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]]) == 0:
    #             image.append(np.expand_dims(temp, axis=0))
    #             overlapping_annotations.append(annotation)
    #     rectified_annotations = []
    #     rectified_masks = []
    #     for ann, img in zip(overlapping_annotations, image):
    #         # print(ann['bbox'])
    #         ann['bbox'] = [ann['bbox'][0] - patch_bbox[0], ann['bbox'][1] - patch_bbox[1],
    #                        ann['bbox'][2] , ann['bbox'][3]]
    #         x1, y1, x2, y2 = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]
    #         temp_box = [x1, y1, x2, y2]
    #         temp_box = np.clip(temp_box, 0, 800)
    #         x = temp_box[0]
    #         y = temp_box[1]
    #         w = temp_box[2] - temp_box[0]
    #         h = temp_box[3] - temp_box[1]
    #         ann['bbox'] = [x, y, w, h]
    #         if w*h > 0:
    #             rectified_annotations.append(ann)
    #             rectified_masks.append(img)
    #     try:
    #         mask = np.stack(rectified_masks, axis=0)
    #         mask = mask[:, :, patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]]
    #     except:
    #         mask = np.zeros((1, 1, 800, 800))
    #     # Show the mask
    #     # plt.figure(figsize=(10, 10))
    #     # t = mask[:, 0, :,:]
    #     # t = np.sum(t, axis=0)
    #     # plt.imshow(t[patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]])
    #     # print(overlapping_annotations)
    #     return rectified_annotations, mask

    def __getitem__(self, idx):
        image_index = self.index_list[idx]
        # print(image_index)
        image_data = self.coco_data.loadImgs(image_index)[0]
        annotation_ids = self.coco_data.getAnnIds(imgIds=image_index)
        annotations = self.coco_data.loadAnns(annotation_ids)
        # if len(annotations) == 0:
        #     next_idx = (idx + 1) % len(self.index_list)
        #     return self.__getitem__(next_idx)
        hr_image_path = os.path.join(self.data_dir, "HR" , image_data['file_name'])
        lr_image_path = os.path.join(self.data_dir, "LR" , image_data['file_name'])
        mask_path = os.path.join(self.mask_dir, image_data['file_name'].split('.')[0] + '.pt')
        # print(image_path)
        # image = cv2.imread(image_path, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hr = Image.open(hr_image_path).convert('RGB')
        image_hr  = np.array(image_hr)

        image_lr = Image.open(lr_image_path).convert('RGB')
        image_lr  = np.array(image_lr)

        mask = self._get_mask(mask_path)
        # image = np.pad(image, ((200, 200), (200, 200), (0, 0)), 
        #                mode='constant', constant_values=0)
        
        ## NOTE: First two coordinates are x, y, need to add 200 to each
        ## NOTE: All annotation coordinates need to be updated (add 200 to all)
        # for annotation in annotations:
        #     annotation['bbox'][0] += 200
        #     annotation['bbox'][1] += 200
        # patch_x, patch_y = patch_index[0] * self.patch_height, patch_index[1] * self.patch_width
        # patch_bbox = [patch_y, patch_x, patch_y + self.patch_height, patch_x + self.patch_width]
        # overlapping_annotations, mask = self._generate_mask(annotations, patch_bbox, mask_path)
        # patch_img = image[patch_x:patch_x + self.patch_height, patch_y:patch_y + self.patch_width, :]
        # print(self.verbose, "********", annotations)
        annotations = self._change_bbox_xyhw_to_xyxy(annotations)
        # patch_img_lr = self.imresize_np(patch_img, 0.25)

        boxes = []
        labels = []
        area = []
        iscrowd = []
        for annotation in annotations:
            box = annotation['bbox']
            if abs(box[0]-box[2]) < 0.1 or abs(box[1]-box[3]) < 0.1:
                # print("Degenrate case")
                continue
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])
            area.append(annotation['area'])
            iscrowd.append(annotation['iscrowd'])

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            area = [1]
            iscrowd = [0]
        target = {}
        target['image'] = image_hr
        target['masks'] = list(mask) if mask is not None else []
        target['image_lq'] = image_lr
        target['bboxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = image_index
        # name = image_data['file_name'].split('.')[0]
        # name = name + '_patch_' + str(patch_index[0]) + '_' + str(patch_index[1]) + '.jpg'
        # lq_path = os.path.join(self.data_dir, name)
        target['LQ_path'] = lr_image_path

        if self.transform is None:
            image, target = self.convert_to_tensor(**target)
            return image, target
        else:
            for i in range(len(target['masks'])):
                target['masks'][i] = target['masks'][i][0]
            transformed = self.transform(**target)
            image, target = self.convert_to_tensor(**transformed)
            return image, target

    def convert_to_tensor(self, **target):
        #convert to tensor
        target['image_lq'] = torch.from_numpy(target['image_lq'].transpose((2, 0, 1)))
        target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
        target['boxes'] = torch.tensor(target['bboxes'], dtype=torch.float32)
        target['masks'] = torch.tensor(target['masks'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target["area"] = torch.tensor(target['area'])
        target["iscrowd"] = torch.tensor(target['iscrowd'])
        target['image_id'] = torch.tensor([target['image_id']])


        image = {}
        image['image_lq'] = target['image_lq']
        image['image'] = target['image']
        image['LQ_path'] = target['LQ_path']

        del target['image_lq']
        del target['image']
        del target['bboxes']
        del target['LQ_path']

        return image, target

