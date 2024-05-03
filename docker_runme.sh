img="conda-image-tracker_mmr:latest"
#-----------------------------------------------------------------------------------------------------------------
#source="./images/international/lviv.avi"
#n_lanes=0
#-----------------------------------------------------------------------------------------------------------------
source="./images/international/UA4L.MP4"
n_lanes=4
#-----------------------------------------------------------------------------------------------------------------
limit=10
#-----------------------------------------------------------------------------------------------------------------
docker run --gpus all -v ${PWD}:/home --workdir /home -it $img python3 main.py --input $source --n_lanes $n_lanes --limit $limit