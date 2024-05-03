import numpy
import cv2
import pandas as pd
import os
import motmetrics as mm
from tqdm import tqdm
from sklearn.cluster import KMeans
import inspect
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_IO
import tools_image
import tools_animation
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_mAP_visualizer
import tools_time_profiler
from CV import tools_vanishing
# ----------------------------------------------------------------------------------------------------------------------
class Pipeliner:
    def __init__(self,folder_out,Detector,Tracker,Tokenizer=None,is_gt_xywh=False,obj_types_id=[],n_lanes=None,limit=None):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)
        self.is_gt_xywh = is_gt_xywh
        self.obj_types_id = obj_types_id
        self.folder_runs = './runs/'
        self.df_summary = None
        self.n_lanes = n_lanes
        self.limit = limit
        self.folder_in = None
        self.folder_out = folder_out
        self.Detector = Detector
        self.Tracker = Tracker
        self.Tokenizer = Tokenizer
        self.V = tools_mAP_visualizer.Track_Visualizer(folder_out,stack_h=False)
        self.df_summary_custom = None
        self.TP = tools_time_profiler.Time_Profiler(verbose=False)
        self.VP = tools_vanishing.detector_VP(folder_out,W=2048,H=1536)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def GPU_health_check(self):
        import torch
        print('GPU:', torch.cuda.get_device_name(0))
        print('Memory Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Memory Cached:', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def name_columns(self,df):
        if df.shape[0] == 0:
            df = pd.DataFrame(columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

        cols = [c for c in df.columns]
        if 'frame_id' not in cols:
            print('Renaming columns!!')
            cols[0] = 'frame_id'
            cols[1] = 'track_id'
            cols[2] = 'x1'
            cols[3] = 'y1'
            cols[4] = 'x2'
            cols[5] = 'y2'
            cols[6] = 'conf'
            df.columns = cols
            df = df.astype({'frame_id': int, 'track_id': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float})

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def update_true(self,df_true,folder_in):
        self.df_true = self.name_columns(df_true)
        self.folder_in = folder_in
        if self.is_gt_xywh:
            self.df_true['x2'] += self.df_true['x1']
            self.df_true['y2'] += self.df_true['y1']
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_pred(self,df_pred):
        self.df_pred = self.name_columns(df_pred)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def match_E(self,df_det,df_track):

        col_start = [c for c in df_det.columns].index('conf') + 1
        emb_C = df_det.shape[1]-col_start
        if emb_C<=0: return df_track

        if df_track.shape[0] > 0:
            df_det['row_id'] = numpy.arange(df_det.shape[0])
            df_track['row_id'] = -1

            for r, row in df_track.iterrows():
                d1,d2,d3,d4 = abs(df_det['x1'] - row['x1']) , abs(df_det['y1'] - row['y1']) , abs(df_det['x2'] - row['x2'])  , abs(df_det['y2'] - row['y2'])
                idx = numpy.argmin(d1+d2+d3+d4)
                val = numpy.min(d1+d2+d3+d4)
                if val < 60:
                    df_track.iloc[r,-1] = idx

            df_track = tools_DF.fetch(df_track,'row_id',df_det,'row_id',[c for c in df_det.columns[col_start:-1]])
            df_det.drop(columns=['row_id'], inplace=True)
            df_track.drop(columns=['row_id'], inplace=True)

        else:
            df_E = pd.DataFrame(numpy.full((df_track.shape[0], emb_C),numpy.nan), columns=df_det.columns[col_start:])
            df_E = df_E.astype({c:str(df_det[c].dtype) for c in df_E.columns.values})
            df_track = pd.concat([df_track,df_E], axis=1)

        return df_track
# ----------------------------------------------------------------------------------------------------------------------
    def __get_detections(self,image,frame_id,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name)
        df_det = pd.DataFrame([])
        image = cv2.imread(image) if isinstance(image, str) else image

        if self.Tracker.__class__.__name__ in ['Tracker_yolo']:
            df_det = self.Tracker.track_detections(df_det, image,frame_id=frame_id,do_debug=do_debug)
            if self.obj_types_id is not None and len(self.obj_types_id) > 0: df_det = df_det[df_det['class_ids'].isin(self.obj_types_id)]
        else:
            df_det = self.Detector.get_detections(image,do_debug=do_debug)

        if self.obj_types_id is not None and len(self.obj_types_id) > 0:
            df_det = df_det[df_det['class_ids'].isin(self.obj_types_id)]

        df_det = tools_DF.add_column(df_det, 'frame_id', frame_id)
        df_det = df_det.reset_index().iloc[:, 1:]
        self.TP.tic(inspect.currentframe().f_code.co_name)

        return df_det
# ----------------------------------------------------------------------------------------------------------------------
    def __get_features(self,filename, df_det,frame_id):
        self.TP.tic(inspect.currentframe().f_code.co_name)
        df_embedding = self.Tokenizer.get_features(filename, df_det, frame_id=frame_id)
        self.TP.tic(inspect.currentframe().f_code.co_name)
        return df_embedding
# ----------------------------------------------------------------------------------------------------------------------
    def __get_tracks(self,filename,df_det,frame_id,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name)

        if self.Tracker is None:
            df_track = tools_DF.add_column(df_det.copy(), 'track_id', -1,pos=1)
        elif self.Tracker.__class__.__name__ not in ['Tracker_yolo']:
            df_track = self.Tracker.track_detections(df_det, filename,frame_id=frame_id, do_debug=do_debug)
        else:
            df_track = df_det.copy()

        if 'frame_id' not in [c for c in df_track.columns]:
            df_track = tools_DF.add_column(df_track, 'frame_id', frame_id)
        df_track = df_track.astype({'frame_id':int,'track_id': int})
        self.TP.tic(inspect.currentframe().f_code.co_name)
        return df_track
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_01_track(self, source, do_debug=False):
        def function(filename, frame_id, do_debug=False):

            mode = 'w' if frame_id == 1 else 'a'
            df_det = self.__get_detections(filename, frame_id=frame_id)

            if self.Tokenizer is not None:
                df_embedding = self.__get_features(filename, df_det,frame_id)
                df_det = pd.concat([df_det, df_embedding], axis=1)

            df_track = self.__get_tracks(filename, df_det, frame_id=frame_id,do_debug=do_debug)
            df_track = self.match_E(df_det, df_track)

            df_det.to_csv(self.folder_out + 'df_det.csv', index=False, header=True if mode == 'w' else False, mode=mode,float_format='%.2f')
            df_track.to_csv(self.folder_out + 'df_track.csv', index=False, header=True if mode == 'w' else False,mode=mode, float_format='%.2f')
            return

        is_video = ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower())

        if is_video:
            vidcap = cv2.VideoCapture(source)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.limit is not None: total_frames = min(total_frames, self.limit)
            for i in tqdm(range(total_frames), total=total_frames, desc=inspect.currentframe().f_code.co_name):
                success, image = vidcap.read()
                if not success: continue
                function(image, frame_id=i + 1, do_debug=do_debug)

            tools_animation.folder_to_video(self.folder_out, self.folder_out + source.split('/')[-1], mask='*.jpg', framerate=24)
            tools_IO.remove_files(self.folder_out, '*.jpg,*.ion')
            vidcap.release()
        else:
            self.folder_in = source
            filenames = tools_IO.get_filenames(source, '*.jpg,*.png')
            if self.limit is not None: filenames = filenames[:self.limit]
            for i,filename in tqdm(enumerate(filenames),total=len(filenames),desc=inspect.currentframe().f_code.co_name):
                function(source + filename, frame_id=i + 1, do_debug=do_debug)

        if os.path.isfile(self.folder_out + 'df_true2.csv'):os.remove(self.folder_out + 'df_true2.csv')
        if os.path.isfile(self.folder_out + 'df_pred2.csv'):os.remove(self.folder_out + 'df_pred2.csv')
        self.df_pred = self.name_columns(pd.read_csv(self.folder_out + 'df_track.csv', sep=','))
        self.TP.stage_stats(self.folder_out + 'time_profile.csv')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_02_calc_benchmarks_quick(self, df_true, df_pred, iou_th=0.5):

        df_true = self.name_columns(df_true)
        df_pred = self.name_columns(df_pred)
        df_pred['x2'] -= df_pred['x1']
        df_pred['y2'] -= df_pred['y1']

        acc = mm.MOTAccumulator(auto_id=False)

        for frame in range(df_true.shape[0]):
            vals_fact = df_true[df_true.iloc[:, 0] == frame+1].iloc[:, 1:6].values
            vals_pred = df_pred[df_pred.iloc[:, 0] == frame+1].iloc[:, 1:6].values
            C = mm.distances.iou_matrix(vals_fact[:, 1:], vals_pred[:, 1:], max_iou=iou_th)
            acc.update(vals_fact[:, 0].astype('int').tolist(), vals_pred[:, 0].astype('int').tolist(), C,frameid=frame+1)

        mh = mm.metrics.create()
        df_summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 'num_objects',
                                              'mostly_tracked', 'partially_tracked', 'mostly_lost',
                                              'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
                                              'mota', 'motp'], name='acc')
        self.df_summary = df_summary.map("{0:.2f}".format)
        print(tools_DF.prettify(self.df_summary.T, showindex=True))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_02_calc_benchmarks_custom(self,iou_th = 0.5):

        df_true2, df_pred2 = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th,from_cache=True)
        ths = df_true2['conf_pred'].unique()
        TPs_det,FNs_det,FPs_det,F1s_det = [],[],[],[]
        TPs_ID,FNs_ID,FPs_ID,F1s_ID = [],[],[],[]

        ths = numpy.sort(ths[~numpy.isnan(ths)])
        for th in ths:
            TPs_det.append(df_true2[(df_true2['conf_pred'] >= th) & (df_true2['pred_row'] != -1)].shape[0])
            FNs_det.append(df_true2.shape[0] - TPs_det[-1])
            FPs_det.append(df_pred2[(df_pred2['conf'] >= th) & (df_pred2['true_row'] == -1)].shape[0])
            F1s_det.append(2 * TPs_det[-1] / (2 * TPs_det[-1] + FPs_det[-1] + FNs_det[-1]) if TPs_det[-1] + FPs_det[-1] + FNs_det[-1] > 0 else 0)

            TPs_ID.append(df_true2[(df_true2['conf_pred'] >= th) & (df_true2['pred_row'] != -1) & (df_true2['IDTP']==True)].shape[0])
            FNs_ID.append(df_true2.shape[0] - TPs_ID[-1])
            FPs_ID.append(df_pred2[(df_pred2['conf'] >= th) & (df_pred2['IDTP']==False)].shape[0])
            F1s_ID.append(2 * TPs_ID[-1] / (2 * TPs_ID[-1] + FPs_ID[-1] + FNs_ID[-1]) if TPs_ID[-1] + FPs_ID[-1] + FNs_ID[-1] > 0 else 0)

        self.V.plot_f1_curve(F1s_det, ths, filename_out=self.folder_out + 'F1_det.png')
        self.V.plot_precision_recall(numpy.array(TPs_det) / (numpy.array(TPs_det) + numpy.array(FPs_det)), numpy.array(TPs_det) / (numpy.array(TPs_det) + numpy.array(FNs_det)),filename_out=self.folder_out + 'PR_det.png',iuo_th=iou_th)
        self.V.plot_f1_curve(F1s_ID, ths, filename_out=self.folder_out + 'F1_ID.png')
        self.V.plot_precision_recall(numpy.array(TPs_ID) / (numpy.array(TPs_ID) + numpy.array(FPs_ID)), numpy.array(TPs_ID) / (numpy.array(TPs_ID) + numpy.array(FNs_ID)),filename_out=self.folder_out + 'PR_ID.png',iuo_th=iou_th)

        idx_best = numpy.argmax(F1s_det)
        precision_det = TPs_det[idx_best] / (TPs_det[idx_best] + FPs_det[idx_best]) if TPs_det[idx_best] + FPs_det[idx_best] > 0 else 0
        recall_det = TPs_det[idx_best] / (TPs_det[idx_best] + FNs_det[idx_best]) if TPs_det[idx_best] + FNs_det[idx_best] > 0 else 0
        df_det = pd.DataFrame({'TP':[TPs_det[idx_best]],'FP':[FPs_det[idx_best]],'FN':[FNs_det[idx_best]],'Pr':precision_det,'Rc':recall_det,'F1':[F1s_det[idx_best]],'th':ths[idx_best]}).T
        df_det = df_det.map("{0:.2f}".format)
        dct_names = dict(zip([c for c in df_det.columns], ['Det @%.2f' % iou for iou in [iou_th]]))
        df_det = df_det.rename(columns=dct_names)

        idx_best = numpy.argmax(F1s_ID)
        precision_ID = TPs_ID[idx_best] / (TPs_ID[idx_best] + FPs_ID[idx_best]) if TPs_ID[idx_best] + FPs_ID[idx_best] > 0 else 0
        recall_ID = TPs_ID[idx_best] / (TPs_ID[idx_best] + FNs_ID[idx_best]) if TPs_ID[idx_best] + FNs_ID[idx_best] > 0 else 0
        df_ID = pd.DataFrame({'TP':[TPs_ID[idx_best]],'FP':[FPs_ID[idx_best]],'FN':[FNs_ID[idx_best]],'Pr':precision_ID,'Rc':recall_ID,'F1':[F1s_ID[idx_best]],'th':ths[idx_best]}).T
        df_ID = df_ID.map("{0:.2f}".format)
        dct_names = dict(zip([c for c in df_ID.columns], ['ID @%.2f' % iou for iou in [iou_th]]))
        df_ID = df_ID.rename(columns=dct_names)

        self.df_summary_custom = pd.concat([df_det,df_ID],axis=1)
        self.df_summary_custom.to_csv(self.folder_out + 'df_summary.csv', index=True)

        print(tools_DF.prettify(self.df_summary_custom, showindex=True))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_04_visualize_sequence(self,iou_th=0.5, conf_th=0.50, use_IDTP=False):
        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=iou_th)
        self.V.draw_sequence(self.folder_in,df_true_rich, df_pred_rich, conf_th=conf_th, use_IDTP=use_IDTP)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_04a_confusion_matrix(self,iou_th=0.5, conf_th=0.50):

        dct_map_true = dict(zip(sorted(self.df_true['track_id'].unique()),range(self.df_true['track_id'].unique().shape[0])))
        dct_map_pred = dict(zip(sorted(self.df_pred['track_id'].unique()),range(self.df_pred['track_id'].unique().shape[0])))

        conf_mat = numpy.zeros((self.df_true['track_id'].unique().shape[0], self.df_pred['track_id'].unique().shape[0]))
        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred,iou_th=iou_th)

        for track_id in df_true_rich.track_id.unique():
            df = df_true_rich[df_true_rich['track_id'] == track_id]
            df_agg = tools_DF.my_agg(df,cols_groupby=['track_id_pred'],cols_value=['frame_id'],aggs=['count'])
            A = df_agg.dropna().values.astype('int')
            for a in A:
                conf_mat[dct_map_true[track_id],dct_map_pred[a[0]]] = a[1]

        df_conf = pd.DataFrame(conf_mat,columns=dct_map_pred.keys(),index=dct_map_true.keys())
        df_conf = df_conf.fillna(0)
        df_conf = self.order_corr_matrix(df_conf)

        self.V.draw_confusion_matrix(self.folder_in,df_true_rich, df_pred_rich,df_conf)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_05_visualize_tracks_simple(self,conf_th = 0.50):
        self.V.draw_stacked_simple(self.folder_in, self.df_true, self.df_pred, conf_th=conf_th)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_06_visualize_tracks_RAG(self,conf_th = 0.50,use_IDTP=False):
        df_true_rich, df_pred_rich = self.V.B.calc_hits_stats_iou(self.df_true, self.df_pred)
        self.V.draw_stacked_RAG(self.folder_in, df_true_rich, df_pred_rich, conf_th=conf_th, use_IDTP=use_IDTP)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_08_scan(self,folder_images,df):
        filenames = tools_IO.get_filenames(folder_images, '*.jpg,*.png')
        df_filenames = pd.DataFrame({'frame_id': numpy.arange(1, 1 + len(filenames)), 'filename': filenames})
        C = 20
        R = 1+len(filenames)//C
        w,h = 92,92

        for obj_id in tqdm(df.track_id.unique(), total=df.track_id.unique().shape[0], desc=inspect.currentframe().f_code.co_name):
            im_scan = numpy.full((R * h, C * w, 3), 32, dtype=numpy.uint8)
            df_obj = df[df['track_id'] == obj_id]
            for frame_id in df.frame_id.unique():
                df_frame = df_obj[df_obj['frame_id'] == frame_id]
                if df_frame.shape[0] == 0: continue
                filename = df_filenames[df_filenames['frame_id'] == frame_id].iloc[0].iloc[1]
                image = cv2.imread(folder_images + filename)
                for r in range(df_frame.shape[0]):
                    rect = df_frame.iloc[r][['x1','y1','x2','y2']].values.astype(int)
                    image_crop = image[rect[1]:rect[3], rect[0]:rect[2]]
                    if image_crop.shape[0]*image_crop.shape[1] == 0: continue
                    image_crop = tools_image.smart_resize(image_crop, w, h)
                    im_scan = tools_image.put_image(im_scan,image_crop,(frame_id//C)*h,(frame_id%C)*w)

            cv2.imwrite(self.folder_out + 'scan_%03d.png' % obj_id, im_scan)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_09_profiles(self, source):
        def get_image(source,df_filenames,frame_id):
            if 'filename' in df_filenames.columns:
                filename = df_filenames[df_filenames['frame_id'] == frame_id].iloc[0].iloc[1]
                image = cv2.imread(source + filename)
            else:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                success, image = vidcap.read()
            return image

        is_video = ('mp4' in source) or ('avi' in source)
        if is_video:
            vidcap = cv2.VideoCapture(source)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            df_filenames = pd.DataFrame({'frame_id': numpy.arange(1, 1+total_frames)})
        else:
            filenames = tools_IO.get_filenames(source, '*.jpg,*.png')
            df_filenames = pd.DataFrame({'frame_id': numpy.arange(1, 1 + len(filenames)), 'filename': filenames})

        tools_IO.remove_folders(self.folder_out)

        for obj_id in tqdm(self.df_pred['track_id'].unique(), total=self.df_pred['track_id'].unique().shape[0],desc=inspect.currentframe().f_code.co_name):
            os.mkdir(self.folder_out + 'profile_%03d' % obj_id)
            df_repr = self.df_pred[self.df_pred['track_id'] == obj_id]
            for r in range(df_repr.shape[0]):
                frame_id = df_repr.iloc[r]['frame_id']
                image = get_image(source,df_filenames,frame_id)
                rect = df_repr.iloc[r][['x1', 'y1', 'x2', 'y2']].values.astype(int)
                confidence = df_repr.iloc[r]['conf']
                confidence = 0 if numpy.isnan(confidence) else confidence
                image_crop = image[rect[1]:rect[3], rect[0]:rect[2]]
                cv2.imwrite(self.folder_out + '/profile_%03d/'% obj_id + '%06d.png' % (frame_id) , image_crop)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_10_get_background_image(self, source, filename_out=None):
        image_clear_bg = self.V.remove_bg(source, self.df_pred)
        if filename_out is not None:
            cv2.imwrite(self.folder_out + filename_out, image_clear_bg)
        return image_clear_bg
# ----------------------------------------------------------------------------------------------------------------------
    def get_VP(self,image_debug=None):
        lines = []
        obj_ids = self.df_pred['track_id'].unique()
        minY = self.df_pred['y1'].min()
        maxY = self.df_pred['y1'].max()
        margin = (maxY-minY)*0.1
        minY+= margin
        maxY-= margin

        for obj_id in obj_ids:
            det_local = self.df_pred[(self.df_pred['track_id']==obj_id) & (self.df_pred['y1']>minY) & (self.df_pred['y2']<maxY)]
            if det_local.shape[0] ==0:continue
            det_local = det_local.sort_values(by='frame_id').iloc[[0, -1]]
            rects_det = det_local[['x1', 'y1', 'x2', 'y2']].values
            cx = rects_det.reshape((-1, 4))[:, [0, 2]].mean(axis=1)
            cy = rects_det.reshape((-1, 4))[:, [1, 3]].mean(axis=1)
            line = [cx[0], cy[0], cx[1], cy[1]]
            lines.append(line)

        point_van_xy_ver, lines_vp_ver = self.VP.get_vp(self.VP.reshape_lines_as_paired(lines),image_debug=image_debug)
        return point_van_xy_ver, lines_vp_ver
# ----------------------------------------------------------------------------------------------------------------------
    def enrich_lane_number(self,h_ipersp,n_lanes):

        self.df_pred['cx'] = (self.df_pred['x1'] + self.df_pred['x2']) / 2
        self.df_pred['cy'] = (self.df_pred['y1'] + self.df_pred['y2']) / 2
        self.df_pred['bev_x'] = 0

        for r in range(self.df_pred.shape[0]):
            center = numpy.array([[[self.df_pred.at[r,'cx'], self.df_pred.at[r,'cy']]]], dtype=numpy.float32)
            point_BEV = cv2.perspectiveTransform(center, h_ipersp).reshape((-1, 2))
            self.df_pred.at[r,'bev_x'] = point_BEV[0][0]

        kmeans = KMeans(n_clusters=n_lanes)
        kmeans.fit(self.df_pred['bev_x'].values.reshape(-1, 1))
        labels = kmeans.labels_
        df_xxx = pd.DataFrame({'bev_x': self.df_pred['bev_x'], 'label': labels})
        df_a0 = tools_DF.my_agg(df_xxx,cols_groupby=['label'], cols_value=['bev_x'], aggs=['mean']).sort_values(by='bev_x')
        df_a = pd.DataFrame({'lane_num': range(df_a0.shape[0]), 'bev_x': df_a0['bev_x']})
        self.df_pred['lane_number'] = self.df_pred['bev_x'].apply(lambda x: numpy.argmin(abs(df_a['bev_x'] - x)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_11_draw_dashboard(self, source,font_size=24):
        is_video = ('mp4' in source.lower()) or ('avi' in source.lower()) or ('mkv' in source.lower())
        if is_video:
            vidcap = cv2.VideoCapture(source)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.limit is not None: total_frames = min(total_frames, self.limit)
            for i in tqdm(range(total_frames), total=total_frames, desc='Extracting frames'):
                success, image = vidcap.read()
                cv2.imwrite(self.folder_out + 'frame_%06d.jpg' % i, image)
                if not success: continue
            vidcap.release()
            source = self.folder_out

        filenames = tools_IO.get_filenames(source, '*.jpg')[:1]
        H, W = cv2.imread(source + filenames[0]).shape[:2]

        if self.n_lanes is not None:
            image_bg = self.pipe_10_get_background_image(source)
            fov_x_deg = 15
            fov_y_deg = fov_x_deg * image_bg.shape[0] / image_bg.shape[1]
            self.VP.H, self.VP.W = image_bg.shape[:2]
            image_debug_trj = None  # image_debug_trj = self.V.draw_trajectories(self.df_pred)
            point_van_xy_ver, lines_vp_ver  = self.get_VP(image_debug_trj)
            image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.VP.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg, point_van_xy_ver, do_rotation=False)
            W+=(int(image_BEV.shape[1] * image_bg.shape[0] / image_BEV.shape[0]))
            self.enrich_lane_number(h_ipersp,self.n_lanes)
        else:
            image_BEV = None
            h_ipersp = None
            self.df_pred['lane_number'] = 0

        image_time_lapse = self.draw_time_lapse(self.df_pred, H=int(H * 0.1),W=W,font_size=font_size)
        self.draw_dashboard(source, self.df_pred, image_BEV=image_BEV, h_ipersp=h_ipersp,image_time_lapse=image_time_lapse)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipe_12_make_video(self,source):
        filename_out = source.split('/')[-1]
        if filename_out == '':
            filename_out = source.split('/')[-2]

        filename_out = filename_out.split('.')[0] + '.mp4'
        tools_animation.folder_to_video_simple(self.folder_out, self.folder_out + filename_out, mask='*.jpg', framerate=24)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_run(self):
        folder_name = tools_IO.get_next_folder_out(self.folder_runs)
        tools_IO.copy_folder(self.folder_out,folder_name,list_of_masks='*.*')
        df_metadata = pd.DataFrame({'detector': [self.Detector.__class__.__name__], 'tracker': [self.Tracker.__class__.__name__], 'tokenizer': [self.Tokenizer.__class__.__name__],'folder_in': [self.folder_in]})
        df_metadata.to_csv(folder_name + 'df_metadata.csv', index=False)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def order_corr_matrix(self,df_Q):
        idx = numpy.argsort(df_Q.sum(axis=1).values)
        df_Q = df_Q.iloc[idx[::-1],:]
        idx = numpy.argsort(df_Q.sum(axis=0).values)
        df_Q = df_Q.iloc[:,idx[::-1]]

        return df_Q
# ----------------------------------------------------------------------------------------------------------------------
    def extract_objects(self,folder_in):

        def process_image(image,frame_id):
            df_pred = self.Detector.get_detections(image, do_debug=True)
            if df_pred is None: return
            for r in range(df_pred.shape[0]):
                rect = df_pred.iloc[r].iloc[1:5].values.astype(int)
                image = image[rect[1]:rect[3], rect[0]:rect[2]]
                obj_type = int(df_pred.iloc[r].iloc[0])
                confidence = int(df_pred.iloc[r].iloc[5] * 100)
                filename_out = 'im_%02d_%03d_%03d.png' % (obj_type, frame_id, r)
                cv2.imwrite(self.folder_out + filename_out, image)

                mode = 'a+' if os.path.exists(self.folder_out + 'descript.ion') else 'w'
                with open(self.folder_out + "descript.ion", mode=mode) as f_handle:
                    f_handle.write("%s %s\n" % (filename_out, '%03d' % confidence))
            return

        tools_IO.remove_files(self.folder_out, '*.jpg,*.ion')
        is_video = ('mp4' in folder_in) or ('avi' in folder_in)

        if is_video:
            vidcap = cv2.VideoCapture(folder_in)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in tqdm(range(total_frames), total=total_frames, desc=inspect.currentframe().f_code.co_name):
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, image = vidcap.read()
                process_image(image,i)
        else:
            filenames = tools_IO.get_filenames(folder_in, '*.jpg')
            for i, filename in tqdm(enumerate(filenames), total=len(filenames), desc=inspect.currentframe().f_code.co_name):
                image = cv2.imread(folder_in+filename)
                process_image(image,i)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_conf_th(self):
        th = float(self.df_summary_custom.iloc[-1].iloc[-1]) if self.df_summary_custom is not None else 0.7
        return th
# ----------------------------------------------------------------------------------------------------------------------
    def draw_dashboard(self, folder_in_images, df_pred, h_ipersp=None, image_BEV=None,image_time_lapse=None):

        filenames = tools_IO.get_filenames(folder_in_images, '*.jpg,*.png')
        if self.limit is not None: filenames = filenames[:self.limit]
        colors80 = tools_draw_numpy.get_colors(80, shuffle=True)
        font_size = 64
        transperency = 0.5
        w = 12

        for i, filename in tqdm(enumerate(filenames), total=len(filenames), desc=inspect.currentframe().f_code.co_name):
            frame_id = i + 1
            if not os.path.isfile(folder_in_images + filename): continue

            image = cv2.imread(folder_in_images + filename)
            image = tools_image.desaturate(image,level=0.25)

            det = df_pred[(df_pred['frame_id']<=frame_id) & (df_pred['track_id'] >= 0)]
            if det.shape[0]>0:
                for obj_id in det['track_id'].unique():
                    color = colors80[(obj_id - 1) % 80]
                    color_fg = (0, 0, 0) if 10 * color[0] + 60 * color[1] + 30 * color[2] > 100 * 128 else (255, 255, 255)
                    det_local = det[det['track_id']==obj_id].sort_values(by=['frame_id'],ascending=False)
                    if det_local['frame_id'].iloc[0] != frame_id:continue
                    points = det_local[['x1', 'y1', 'x2', 'y2']].values
                    cx = points.reshape((-1, 4))[:, [0, 2]].mean(axis=1).reshape((-1, 1))
                    cy = points.reshape((-1, 4))[:, [1, 3]].mean(axis=1).reshape((-1, 1))
                    centers = numpy.concatenate((points[:, [0, 2]].mean(axis=1).reshape((-1, 1)),points[:, [1, 3]].mean(axis=1).reshape((-1, 1))), axis=1).astype(float).reshape((-1, 2))

                    if det_local.shape[0]>=2:
                        lines = numpy.concatenate((centers[:-1], centers[1:]), axis=1).astype(int)
                        image = tools_draw_numpy.draw_lines(image, lines, color=color, w=w, transperency=transperency,antialiasing=False)

                    image = tools_draw_numpy.draw_text(image, str(obj_id), (int(cx[0]), int(cy[0])), color_fg=color_fg, clr_bg=color,font_size=font_size,hor_align='center',vert_align='center')

                    if 'mmr_type' in det_local.columns and 'conf_mmr' in det_local.columns and 'model_color' in det_local.columns and 'lp_symb' in det_local.columns:
                        mmr_type = det_local['mmr_type'].iloc[0]
                        #conf_mmr = det_local['conf_mmr'].iloc[0]
                        model_color = det_local['model_color'].iloc[0]
                        LP_symb = det_local['lp_symb'].iloc[0]
                        image = tools_draw_numpy.draw_text(image, str(mmr_type) + ' ' +  str(model_color) , (int(cx[0]), int(cy[0]+0.75*font_size)), color_fg=(255,255,255),alpha_transp=0.75,clr_bg=(64,64,64), font_size=0.75*font_size, hor_align='center',vert_align='center')
                        image = tools_draw_numpy.draw_text(image, str(LP_symb) , (int(cx[0]), int(cy[0]+1.5*font_size)), color_fg=(255,255,255),clr_bg=(64,64,64), alpha_transp=0.75,font_size=0.75*font_size, hor_align='center',vert_align='center')

            det = df_pred[(df_pred['frame_id'] == frame_id) & (df_pred['track_id'] >= 0)]

            if image_BEV is not None:
                image_BEV_local = image_BEV.copy()
                if det.shape[0]>0:
                    for obj_id in det['track_id'].unique():
                        color = colors80[(obj_id - 1) % 80]
                        color_fg = (0, 0, 0) if 10 * color[0] + 60 * color[1] + 30 * color[2] > 100 * 128 else (255, 255, 255)
                        det_local = det[det['track_id'] == obj_id].sort_values(by=['frame_id'], ascending=False)
                        points = det_local[['x1', 'y1', 'x2', 'y2']].values
                        label = str(obj_id)
                        centers = numpy.concatenate((points[:,[0,2]].mean(axis=1).reshape((-1, 1)),points[:,[1,3]].mean(axis=1).reshape((-1, 1))),axis=1).astype(float).reshape((-1, 1, 2))
                        points_BEV = cv2.perspectiveTransform(centers,h_ipersp).reshape((-1, 2))
                        image_BEV_local = tools_draw_numpy.draw_text(image_BEV_local, label, (int(points_BEV[:,0]), int(points_BEV[:,1])), color_fg=color_fg,clr_bg=color, font_size=font_size, hor_align='center',vert_align='center')

                image_BEV_local = cv2.resize(image_BEV_local, (int(image_BEV_local.shape[1] * image.shape[0] / image_BEV_local.shape[0]),image.shape[0]), interpolation=cv2.INTER_NEAREST)
                image = numpy.concatenate((image, image_BEV_local), axis=1)


            image_time_lapse_local = cv2.resize(image_time_lapse, (image.shape[1],int(image_time_lapse.shape[0]*image.shape[1]/image_time_lapse.shape[1])), interpolation=cv2.INTER_NEAREST)
            x = int(image_time_lapse.shape[1] * frame_id / df_pred['frame_id'].max())
            image_time_lapse_local[:,x-1:x+1] = 128
            cv2.imwrite(self.folder_out + filename.split('.')[0] + '.jpg',numpy.concatenate((image, image_time_lapse_local), axis=0))

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_time_lapse(self,df_pred,W,H,font_size=24,conf_th=0.1):

        colors80 = tools_draw_numpy.get_colors(80, shuffle=True)
        image = numpy.full((H, W, 3), 32, dtype=numpy.uint8)

        df_pos = tools_DF.my_agg(df_pred,cols_groupby=['track_id'],cols_value=['frame_id'],aggs=['mean'],list_res_names=['position'])
        df_pos = tools_DF.fetch(df_pos,'track_id',df_pred,'track_id',col_value=['lane_number'])
        df_pos = df_pos.sort_values(by=['lane_number','position'],ascending=[True,True])
        df_pos['sub_lane'] = -2*2 + 2*(numpy.arange(df_pos.shape[0]) % 4)

        L = df_pos['lane_number'].max()
        step = float(H/(L + 1))
        #image[[int(step*l) for l in range(L+1)],:]=64

        for r in range(df_pos.shape[0]):
            obj_id = df_pos['track_id'].iloc[r]
            col = colors80[(obj_id - 1) % 80]
            Y = step / 2 + step * df_pos['lane_number'].iloc[r]
            x_start = (df_pred[df_pred['track_id']==obj_id]['frame_id'].min())*(W-1)/df_pred['frame_id'].max()
            x_stop  = (df_pred[df_pred['track_id']==obj_id]['frame_id'].max())*(W-1)/df_pred['frame_id'].max()
            delta = df_pos['sub_lane'].iloc[r]
            image = tools_draw_numpy.draw_line(image, Y+delta,x_start, Y+delta, x_stop, color_bgr=col, antialiasing=True)

        for r in range(df_pos.shape[0]):
            obj_id = df_pos['track_id'].iloc[r]
            col = colors80[(obj_id - 1) % 80]
            color_fg = (0, 0, 0) if 10 * col[0] + 60 * col[1] + 30 * col[2] > 100 * 128 else (255, 255, 255)
            X = df_pos['position'].iloc[r]*(W-1)/df_pred['frame_id'].max()
            Y = step / 2 + step * df_pos['lane_number'].iloc[r]
            image = tools_draw_numpy.draw_text(image, obj_id.astype(str), (int(X), int(Y)), color_fg=color_fg,clr_bg=col, font_size=font_size, hor_align='center', vert_align='center')


        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_trajectories(self,df_pred):
        points = df_pred[['x1', 'y1', 'x2', 'y2']].values
        centers = numpy.concatenate((points[:, [0, 2]].mean(axis=1).reshape((-1, 1)),points[:, [1, 3]].mean(axis=1).reshape((-1, 1))), axis=1).astype(float).reshape((-1, 2))
        image = numpy.full((int(centers[:, 1].max()), int(centers[:, 0].max()), 3), 32, dtype=numpy.uint8)
        colors = (0,0,200)
        if 'lane_number' in df_pred.columns:
            colors80 = tools_draw_numpy.get_colors(80, shuffle=True)
            colors = [colors80[(l - 1) % 80] for l in df_pred['lane_number'].values]

        image = tools_draw_numpy.draw_points(image,centers,color=colors,w=4)
        return image
# ----------------------------------------------------------------------------------------------------------------------