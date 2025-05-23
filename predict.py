import sys
import os

# Lấy đường dẫn tuyệt đối của thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
# Thêm đường dẫn repository vào sys.path
sys.path.append(os.path.join(current_dir, 'EfficientDET/Yet-Another-EfficientDet-Pytorch'))

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 0
force_input_size = None  # set None to use default size
img_path = '/content/Yet-Another-EfficientDet-Pytorch/datasets/aaa22_jpg.rf.17b215afd168e58de3a62cb2b1625d32.jpg' #thay bang duong dan toi anh that

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = [
    'Cam bam coi',
    'Cam di nguoc chieu',
    'Cam do xe',
    'Cam do xe ngay chan',
    'Cam do xe ngay le',
    'Cam dung xe va do xe',
    'Cam oto re trai',
    'Cam quay dau',
    'Cam re phai',
    'Cam re trai',
    'Cam xe may 2 banh',
    'Cho ngoac nguy hiem phia ban trai',
    'Cho ngoac nguy hiem phia ben phai',
    'Duong giao nhau',
    'Duong nguoi di bo cat ngang',
    'Giao nhau voi duong khong uu tien',
    'Giao nhau voi duong uu tien',
    'Huong di thang phai theo',
    'Noi giao nhau theo vong xuyen',
    'Toc do toi da 40 km-h',
    'Tre em qua duong',
    'toc do toi da 60 km-h',
    'toc do toi da 80 km-h'
]


# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

model.load_state_dict(torch.load('efficientdet-d0_59_3400.pth')) #thay bang duong dan toi model that
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


with torch.no_grad():
    features, regression, classification, anchors = model(x)
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

out = invert_affine(framed_metas, out)

for i in range(len(ori_imgs)):
    if len(out[i]['rois']) == 0:
        continue

    # Tìm index của detection có score cao nhất
    if len(out[i]['scores']) > 0:
        max_score_idx = np.argmax(out[i]['scores'])

        ori_imgs[i] = ori_imgs[i].copy()
        (x1, y1, x2, y2) = out[i]['rois'][max_score_idx].astype(int)
        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)

        obj = obj_list[out[i]['class_ids'][max_score_idx]]
        score = float(out[i]['scores'][max_score_idx])

        print(f"Highest confidence detection: {obj} with score: {score:.3f}")

        cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        plt.imshow(ori_imgs[i])
        plt.axis('off')
        plt.show()

