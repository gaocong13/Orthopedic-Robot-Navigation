LandmarkAnnot=/home/cong/Research/Spine/CadaverNeedleInjection/landmark_annotation
Device3DLandmark=/home/cong/Research/Spine/CadaverNeedleInjection/meta_data/Device3Dlandmark.fcsv
Device3DBB=/home/cong/Research/Spine/CadaverNeedleInjection/meta_data/Device3Dbb.fcsv
ExpTXT=/home/cong/Research/Spine/CadaverNeedleInjection/expID.txt
ImageDCM=/home/cong/Research/Spine/CadaverNeedleInjection/RealXraydcm
Output=/home/cong/Research/Spine/CadaverNeedleInjection/output
xreg-spine-device-regi-singleview ${LandmarkAnnot} ${Device3DLandmark} ${Device3DBB} ${ExpTXT} ${ImageDCM} ${Output} -v
