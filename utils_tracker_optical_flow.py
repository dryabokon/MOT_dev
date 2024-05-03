import numpy
import cv2
import os
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_optical_flow
# ----------------------------------------------------------------------------------------------------------------------
class Tracker_optical_flow:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out = folder_out
        self.is_initialized = False
        self.OF = None
        return
# ----------------------------------------------------------------------------------------------------------------------
    def track_detections(self,df_det,filename_in,frame_id=None,do_debug=True):

        image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in

        if not self.is_initialized:
            self.OF = tools_optical_flow.OpticalFlow_LucasKanade(image, folder_out=self.folder_out)
            self.OF.remove_keypoints(df_det[['x1', 'y1', 'x2', 'y2']].values)
            self.is_initialized = True


        self.OF.evaluate_flow(image)

        if do_debug:
            cv2.imwrite(self.folder_out + filename_in.split('/')[-1],self.OF.draw_current_frame())

        self.OF.next_step()
        return pd.DataFrame([])
# ----------------------------------------------------------------------------------------------------------------------