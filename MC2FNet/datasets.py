# _RGBD_SOD_ROOT = "<rgbdsod root>"
# _RGBT_SOD_ROOT = "<rgbtsod root>"
_RGBD_SOD_ROOT = "/home/hutao/Tanz_hanzhong_files/RGB-D-T/CAVER-main/rgbdsod root"
_RGBT_SOD_ROOT = "/home/hutao/Tanz_hanzhong_files/RGB-D-T/CAVER-main-2/rgbtsod root"

# RGB-D SOD
LFSD = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/LFSD/Mask", suffix=".png"),
)
NLPR_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NLPR/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NLPR/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NLPR/Mask", suffix=".png"),
    # index_file="datasets/nlpr_train_jw_name_list.lst",
)
NLPR_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NLPR/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NLPR/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NLPR/Mask", suffix=".png"),
)
NJUD_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NJUD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NJUD/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/NJUD/Mask", suffix=".png"),
)
NJUD_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/NJUD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/NJUD/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/NJUD/Mask", suffix=".png"),
)
RGBD135 = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/RGBD135/Mask", suffix=".png"),
)
SIP = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/SIP/Mask", suffix=".png"),
)
SSD = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Depth", suffix=".bmp"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/SSD/Mask", suffix=".png"),
)
STEREO1000 = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/STERE/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/STERE/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/STERE/Mask", suffix=".png"),
)
DUTRGBD_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/DUTLF-Depth/Mask", suffix=".png"),
)
DUTRGBD_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/DUTLF-Depth/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/DUTLF-Depth/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/Trainset/DUTLF-Depth/Mask", suffix=".png"),
)
REDWEBS_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/trainset/Mask", suffix=".png"),
)
REDWEBS_TE = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/Image", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/Depth", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/ReDWeb-S/testset/Mask", suffix=".png"),
)

COME_TR = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/imgs_right", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TR/gt_right", suffix=".png"),
)
COME_TE_E = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-E/GT", suffix=".png"),
)
COME_TE_H = dict(
    image=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/depths", suffix=".png"),
    mask=dict(path=f"{_RGBD_SOD_ROOT}/COME15K/COME-TE/COME-TE-H/GT", suffix=".png"),
)


# RGB-T SOD
VT5000TR = dict(
image=dict(path=f"{_RGBT_SOD_ROOT}/train/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/train/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/train/GT", suffix=".png"),
)
VT5000TE = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/test/VT5000/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/test/VT5000/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/test/VT5000/GT", suffix=".png"),
)
VT1000 = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/test/VT1000/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/test/VT1000/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/test/VT1000/GT", suffix=".png"),
)
VT821 = dict(
    image=dict(path=f"{_RGBT_SOD_ROOT}/test/VT821/RGB", suffix=".jpg"),
    depth=dict(path=f"{_RGBT_SOD_ROOT}/test/VT821/T", suffix=".jpg"),
    mask=dict(path=f"{_RGBT_SOD_ROOT}/test/VT821/GT", suffix=".png"),
)
