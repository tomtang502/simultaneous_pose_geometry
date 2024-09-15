echo "Loading checkpoint for dust3r"
mkdir -p weights
wget -P weights https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth

FILE_ID="1pvZsb8E5u3UcJMeysnaNEtcm_5sB0PuU"
FILE_NAME="sample_data.zip"
gdown --id ${FILE_ID} -O ${FILE_NAME}

mkdir data
unzip ${FILE_NAME} -d data
rm ${FILE_NAME}