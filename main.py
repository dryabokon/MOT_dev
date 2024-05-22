import sys
sys.path.append('./tools')
# ----------------------------------------------------------------------------------------------------------------------
import argparse
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import pipelines
import config_pipe,config_viz
import utils_detector_yolo, utils_detector_LPVD, utils_detector_detectron2,utils_detector_cube_pyr
import utils_tracker_deep_sort,utils_tracker_boxmot, utils_tracker_yolo,utils_tracker_optical_flow
import utils_tokenizer_MMR,utils_tokenizer_cube_pyr
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './output/'
# ----------------------------------------------------------------------------------------------------------------------
def init_detector(args):
    if   args.detector == 'LPVD':       D = utils_detector_LPVD.Detector_LPVD(folder_out)
    elif args.detector == 'yolo':       D = utils_detector_yolo.Detector_yolo(folder_out)
    elif args.detector == 'Detectron2': D = utils_detector_detectron2.Detector_detectron2(folder_out)
    elif args.detector == 'CUBE_PYR':   D = utils_detector_cube_pyr.Detector_Cube_Pyr(folder_out)
    else: D=None
    return D
# ----------------------------------------------------------------------------------------------------------------------
def init_tracker(args):
    if   args.tracker == 'DEEPSORT': T = utils_tracker_deep_sort.Tracker_deep_sort(folder_out)
    elif args.tracker == 'BYTE':     T = utils_tracker_boxmot.Tracker_boxmot(folder_out,algorithm=args.tracker)
    elif args.tracker == 'yolo':     T = utils_tracker_yolo.Tracker_yolo(folder_out)
    elif args.tracker == 'OF':       T = utils_tracker_optical_flow.Tracker_optical_flow(folder_out)
    else: T=None
    return T
# ----------------------------------------------------------------------------------------------------------------------
def init_tokenizer(args):
    if   args.tokenizer == 'MMR': K = utils_tokenizer_MMR.Tokenizer_MMR(folder_out)
    elif args.tokenizer == 'CUBE_PYR': K = utils_tokenizer_cube_pyr.Tokenizer_Cube_Pyr(folder_out)
    else: K=None
    return K
# ----------------------------------------------------------------------------------------------------------------------
def init_pipeline(cnfg,cnf_viz):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', help='path to folder or video',default=cnfg.source)
    parser.add_argument('--output', '-out', help='path to output folder', default=folder_out)
    parser.add_argument('--detector', '-d', help='yolo | Detectron2 |LPVD ', default=cnfg.detector)
    parser.add_argument('--tracker', '-t', help='DEEPSORT | OCSORT | BOTSORT | BYTE', default=cnfg.tracker)
    parser.add_argument('--tokenizer', '-k', help='MMR | NONE', default=cnfg.tokenizer)
    parser.add_argument('--n_lanes', '-l', help='number of lanes: 2,3,4..', default=cnfg.n_lanes)
    parser.add_argument('--start', help='starting frame from the input', default=cnfg.start)
    parser.add_argument('--limit', help='# of frames from the input', default=cnfg.limit)
    args = parser.parse_args()

    source = args.input
    start = int(args.start) if args.start is not None else 0
    limit = int(args.limit) if args.limit is not None else None
    if limit==0: limit=None
    n_lanes = int(args.n_lanes) if args.n_lanes is not None else None
    if n_lanes==0: n_lanes=None

    D = init_detector(args)
    T = init_tracker(args)
    K = init_tokenizer(args)

    P = pipelines.Pipeliner(folder_out, cnfg,cnf_viz,D, T, K)
    return P,source
# ----------------------------------------------------------------------------------------------------------------------
# cv2.imwrite(folder_out + 'background.png', P.get_background_image(source, filename_out='None'))
# ----------------------------------------------------------------------------------------------------------------------
image_BEV, h_ipersp = None,None
# ----------------------------------------------------------------------------------------------------------------------
cnfg_MMR = config_pipe.get_config_MMR()
cnfg_UAZ = config_pipe.get_config_marine_UAZ()
cnfg_BEV_live = config_pipe.get_config_BEV_live()
cnfg_BEV_offline = config_pipe.get_config_BEV_offline()
config_GIS_live = config_pipe.get_config_GIS_live()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_IO.remove_files(folder_out, '*.jpg')
    cnfg_pipe = cnfg_MMR
    cnfg_viz = config_viz.cnfg_viz()

    P,source = init_pipeline(cnfg_pipe, cnfg_viz)

    P.pipe_01_track(source,do_debug=False)
    P.update_pred(folder_out + 'df_track.csv')
    #P.pipe_09_profiles(source)
    P.pipe_11_draw_dashboard_BEV(source)
    P.pipe_12_make_video(source)
    tools_IO.remove_files(folder_out, '*.jpg')

