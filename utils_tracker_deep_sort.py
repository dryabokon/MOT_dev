import numpy
import pandas as pd
import cv2
import os
# ----------------------------------------------------------------------------------------------------------------------
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_deep_sort:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.2)
        self.tracker = Tracker(metric)
        self.colors80 = tools_draw_numpy.get_colors(80, colormap='nipy_spectral', shuffle=True)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,df_det,filename_in,frame_id=None,do_debug=False):

        rects0 = df_det[['x1', 'y1', 'x2', 'y2']].values
        tlwhs0 = [(r[0],r[1],r[2]-r[0],r[3]-r[1]) for r in rects0]
        confs0 = df_det['conf'].values
        features = [[] for i in range(len(tlwhs0))]
        # if df_det.shape[1]>col_start:
        #     features = df_det.iloc[:,col_start:].values

        tlwhs0 = numpy.array(tlwhs0).astype(int)
        self.tracker.predict()
        self.tracker.update([Detection(tlwh, confidence, feature) for tlwh, confidence,feature  in zip(tlwhs0, confs0,features)])

        track_ids = numpy.array([track.track_id for track in self.tracker.tracks if not (not track.is_confirmed() or track.time_since_update > 0)]).astype(int)
        rects = numpy.array([track.to_tlbr().astype(int) for track in self.tracker.tracks if not (not track.is_confirmed() or track.time_since_update > 0)])
        confs = numpy.array([1]*len(track_ids))

        if do_debug:
            if isinstance(filename_in, str):
                image = cv2.imread(filename_in)
                filename_out = (filename_in.split('/')[-1]).split('.')[0] + '.jpg'
            else:
                image = filename_in
                filename_out = 'frame_%06d'%frame_id + '.jpg'

            cv2.imwrite(self.folder_out + filename_out, self.draw_tracks(image, rects.reshape((-1,2,2)), track_ids))

        df_pred = pd.DataFrame(numpy.concatenate((track_ids.reshape((-1, 1)), rects.reshape((-1, 4)), confs.reshape(-1, 1)), axis=1),columns=['track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        return df_pred
# ----------------------------------------------------------------------------------------------------------------------
    def draw_tracks(self,image,rects,track_ids):
        colors = [self.colors80[track_id % 80] for track_id in track_ids]
        image = tools_draw_numpy.draw_rects(tools_image.desaturate(image), rects, colors, labels=track_ids.astype(str), w=2,alpha_transp=0.8)
        return image
# ----------------------------------------------------------------------------------------------------------------------
