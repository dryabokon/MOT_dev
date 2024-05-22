# ----------------------------------------------------------------------------------------------------------------------
class cnfg_base(object):
    source = None
    detector = None
    tracker = None
    tokenizer = None
    n_lanes = 0
    dashboard_simple = False
    start = 0
    limit = 0
    fov_x_deg = 15
    list_of_attributes = []
    image_bg = None
    point_van_xy_ver = None
    crop_rect = [0, 1.0, 0.0, 1.0]
    attributes = []
    tokenizer_mode_simple = True
    stride = 0
# ----------------------------------------------------------------------------------------------------------------------
def get_config_MMR():
    class cnfg(cnfg_base):
        #source = './images/international/KZ.mp4'
        #source = './images/international/UA4L.mp4'
        source = './images/international/CZ.avi'
        detector = 'LPVD'
        tracker = 'BYTE'
        tokenizer = 'MMR'
        n_lanes = 3
        dashboard_simple = False
        start = 0
        limit = 1200
        stride = 3
        attributes = ['mmr_type' , 'conf_mmr' , 'model_color' , 'lp_symb' ]
        tokenizer_mode_simple = False
    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_marine_UAZ():
    class cnfg(cnfg_base):
        #source = './images/marine/UAZ469_v06.mp4'
        #source = './images/marine/UAZ469_v06.mp4'
        #source = './images/marine/UAZ3962_v06.mp4'
        #source = './images/marine/WIN_20240519_08_29_59_Pro.mp4'       #cube and Pyr
        source = './images/marine/WIN_20240519_08_31_59_Pro.mp4'       #empty bg - 2 vehicles
        #source = './images/marine/WIN_20240519_08_37_16_Pro.mp4'        #color bg - 2 vehicles
        #source = '0'

        detector = 'CUBE_PYR'
        tracker = 'DEEPSORT'
        tokenizer = None

        n_lanes = 0
        dashboard_simple = True
        attributes = ['class_name']
        start = 0
        limit = 500
    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_BEV_offline():
    class cnfg(cnfg_base):
        #source = 'https://www.youtube.com/watch?v=71SNlChW_nA'   #CZ2
        source = './images/AEK/71SNlChW_nA.mp4'

        detector = 'yolo'
        tracker = 'DEEPSORT'
        tokenizer = None

        n_lanes = 3
        dashboard_simple = False
        point_van_xy_ver = (1611,119)
        fov_x_deg = 60
        image_bg = './images/AEK/71SNlChW_nA_bg.png'
        crop_rect = [0.6,1.0,0.05,0.95]
        attributes = []
        start = 0
        limit = 0
    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_BEV_live():
    class cnfg(cnfg_base):
        source = 'https://www.youtube.com/watch?v=71SNlChW_nA'   #CZ2

        detector = 'yolo'
        tracker = 'BYTE'
        tokenizer = None

        n_lanes = 3
        dashboard_simple = False
        point_van_xy_ver = (1611,119)
        fov_x_deg = 60
        image_bg = './images/AEK/71SNlChW_nA_bg.png'
        crop_rect = [0.6,1.0,0.05,0.95]
        attributes = []

    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_GIS_live():
    class cnfg(cnfg_base):
        source = 'https://www.youtube.com/watch?v=IG6hIkxq0JU'  #CZ3
        detector = 'yolo'
        tracker = 'BYTE'
        tokenizer = None
    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------