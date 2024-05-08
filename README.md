# Object detection and tracking 
 
You can run the project using the command python main.py followed by the parameters. Here's how to use the parameters:
--input or -i: This parameter is used to specify the path to the folder or video. 
--output or -out: This parameter is used to specify the path to the output folder. 
--detector or -d: This parameter is used to specify the detector to use. The options are 'yolo' or 'LPVD'. The default value is 'LPVD'. 
--tracker: This parameter is used to specify the tracker to use. The options are 'DEEPSORT', 'OCSORT', 'BYTE', 'BOTSORT'. The default value is 'DEEPSORT'. 
--tokenizer or -t: This is an optional parameter used to specify the tokenizer. The options are 'MMR' or 'NONE'. The default value is 'MMR'. 
--n_lanes or -l: This is an optional parameter used to specify the number of lanes. It can be any integer value like 2, 3, 4, etc. Keep it empty to prevent build of the BEV images.
--start: This parameter is used to specify the number of frames from the input. The default value is 120. 
--limit: This parameter is used to specify the limit of frames from the input. The default value is 180.

python main.py --input ./path/to/input --output ./path/to/output --detector yolo --tracker DEEPSORT --tokenizer MMR --n_lanes 3 --start 100 --limit 200


Alternatively you may run the project in docker, see example of the script below

img="conda-image-tracker_mmr:latest"
source="./images/international/lviv.avi"
n_lanes=4
limit=10
docker run --gpus all -v ${PWD}:/home --workdir /home -it $img python3 main.py --input $source --n_lanes $n_lanes --limit $limit --tokenizer None

![alt text](/output/Image.png)
