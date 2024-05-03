import sys
sys.path.append('./tools')
# ----------------------------------------------------------------------------------------------------------------------
import argparse
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import pipelines
import utils_detector_yolo, utils_detector_LPVD
import utils_tracker_deep_sort
import utils_tokenizer_MMR
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './output/'
# ----------------------------------------------------------------------------------------------------------------------
def init_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to folder or video')
    parser.add_argument('--output', '-out', help='path to output folder', default=folder_out)
    parser.add_argument('--detector', '-d', help='yolo | LPVD', default='LPVD')
    parser.add_argument('--tokenizer', '-t', help='MMR | NONE', default='MMR')
    parser.add_argument('--n_lanes', '-l', help='number of lanes: 2,3,4..', default=None)
    parser.add_argument('--limit', help='# of frames from the input', default=None)
    args = parser.parse_args()

    source = args.input

    D = utils_detector_LPVD.Detector_LPVD(folder_out) if args.detector == 'LPVD' else utils_detector_yolo.Detector_yolo(folder_out)
    T = utils_tracker_deep_sort.Tracker_deep_sort(folder_out)
    K = utils_tokenizer_MMR.Tokenizer_MMR(folder_out) if args.tokenizer == 'MMR' else None

    limit = int(args.limit) if args.limit is not None else None
    if limit==0: limit=None
    n_lanes = int(args.n_lanes) if args.n_lanes is not None else None
    if n_lanes==0: n_lanes=None
    P = pipelines.Pipeliner(folder_out, D, T, K,n_lanes=n_lanes, limit=limit)
    return P,source
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out, '*.jpg')

    P,source = init_pipeline()

    P.pipe_01_track(source)
    P.pipe_11_draw_dashboard(source)
    P.pipe_12_make_video(source)

    tools_IO.remove_files(folder_out, '*.jpg')