# settings
TEXT_PROMPT="Audrey Hepburn wearing a tailored blazer, a shirt underneath, straight-cut trousers, and low-heeled shoes."
IMAGE_PROMPT="/path/to/GaussianIP/assets/audrey.png"
CONFIG_PATH="/path/to/GaussianIP/configs/exp.yaml"
LOG_PATH="/path/to/GaussianIP/logs"
CUR_TIME=$(date +"%Y%m%d-%H%M%S")
mkdir -p "$LOG_PATH/$CUR_TIME"

# stage 1: AHDS
echo "Starting stage 1"
CUDA_VISIBLE_DEVICES=0 python launch.py --cur_time "$CUR_TIME" --train system.prompt_processor.prompt="$TEXT_PROMPT" system.guidance.pil_image_faceid_path="${IMAGE_PROMPT}" system.log_path="${LOG_PATH}" system.cur_time="${CUR_TIME}"
echo "Finished stage 1"

# stage 2: View Consistent Refinement
echo "Starting stage 2"
cd /path/to/GaussianIP/threestudio/models/guidance
CUDA_VISIBLE_DEVICES=0 python refine.py --config_path "${CONFIG_PATH}" --log_path "${LOG_PATH}" --cur_time "${CUR_TIME}" --pil_image_path "${IMAGE_PROMPT}" --prompt "$TEXT_PROMPT"
echo "Finished stage 2"

# stage 3: 3D Reconstruction
echo "Starting stage 3"
cd /path/to/GaussianIP
CUDA_VISIBLE_DEVICES=0 python launch.py --cur_time "$CUR_TIME" --train system.cur_time="${CUR_TIME}"
echo "Finished stage 3"
