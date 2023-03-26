
## Dataset
### pose:
- 300wlp (flip and original) 
- biwi
- hpidataset
- 300wlp_flip: randomly rotated 300wlp dataset
- 300wlp_flip2: randomly rotated 300wlp dataset
- biwi_flip: randomly rotated biwi dataset
- china: passport-style photo
### blur:
- blurset
- widerface
## Annotations
Under annotations/ folder are the annotations of combinations of various datasets:
### bs_wf_300wlp:
    blur datasets: blurset, widerface
	pose datasets: 300wlp original(non flip)
	size: 'train': 83289, 'val': 9397
### bs_wf_300wlpFlip
    blur datasets: blurset, widerface
	pose datasets & 300wlp flip:
	size: 'train': 83289, 'val': 9397
### bs_wf_biwi:
    blur datasets: blurset, widerface
	pose datasets: biwi
	size: 'train': 42296, 'val': 4842
### bs_hpi_china:
	blur datasets: blurset(partial)
	pose datasets: hpi, china
	size: 'train': 22059, 'val': 2436
### bs_wf_300wlpRot_biwiRot:
	blur datasets: blurset, widerface
	pose datasets: 300wlp_flip, 300wlp_flip2, biwi_flip, blurset(only blur value>0.83), widerface(only blur value>0.83). In these five datasets, 						euler angles are estimated by PRNet
	size: 'train': 60742, 'val': 6893

## annotations format:
dataset/image_directory,tag,blur,yaw,pitch,roll

- tag: 0--image is for blur detection; 1--for pose estimation.
- blur: blur value for image, ranges from 0 to 1. Higher value incidates better image quality. Blur value > 0.8 have good image quality and little blur. For images used for pose estimation, blur=-1
yaw, pitch, roll: euler angles. For blur detection, these angles equal -1. 
