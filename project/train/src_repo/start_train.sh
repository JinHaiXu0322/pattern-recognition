CURRENT_DIR=$(cd $(dirname $0)/deeplabv3-plus-pytorch; pwd)
echo "${CURRENT_DIR}"
cd "${CURRENT_DIR}"

cp /home/data/1945/*.jpg /project/train/src_repo/deeplabv3-plus-pytorch/VOCdevkit/VOC2007/JPEGImages
cp /home/data/1945/*.png /project/train/src_repo/deeplabv3-plus-pytorch/VOCdevkit/VOC2007/SegmentationClass

python /project/train/src_repo/deeplabv3-plus-pytorch/voc_annotation.py
python /project/train/src_repo/deeplabv3-plus-pytorch/train.py
