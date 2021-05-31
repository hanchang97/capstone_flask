from flask import Flask
from flask_cors import CORS
import os


# flask 가상환경 ver : 3.7.3

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .views import main_views
    app.register_blueprint(main_views.bp)

    main_views.call_Model()
    print('call model complete')
    main_views.call_ypr_Model()
    print('call ypr_model complete')
    main_views.call_face_Model()
    print('call face_model complete')
    main_views.call_eye_Model()
    print('call eye_model complete')

    # focus 플랫폼 전용 폴더를 우분투 서버 내부 경로에 생성한다
    ### 이미 존재하는 경로인지 검사 ###
    focus_main_directory = "C:/FocusHawkEyeMain"
    try:
         if not os.path.exists(focus_main_directory):
             os.makedirs(focus_main_directory)  # 디렉토리 생성 / 그룹 폴더 생성
             print("=== FocusHawkEyeMain folder created ===")
    except OSError:
         print("=== FocusHawkEyeMain folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

     # 웹캠 캡쳐 후 test 이미지가 들어갈 경로
    focus_capture_image_directory = focus_main_directory + '/webCamCapture/temp'
    try:
         if not os.path.exists(focus_capture_image_directory):
             os.makedirs(focus_capture_image_directory)  # 디렉토리 생성 / 웹캠 캡쳐 이미지 폴더 생성
             print("=== FocusHawkEye WebCam Capture folder created ===")
    except OSError:
         print("=== FocusHawkEye WebCam Capture folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

     # face recognition에서 npz 파일 저장 될 경로
    focus_npz_directory = focus_main_directory + '/npzSave'
    try:
         if not os.path.exists(focus_npz_directory):
             os.makedirs(focus_npz_directory)  # 디렉토리 생성 / npz 파일 저장 폴더 생성
             print("=== FocusHawkEye npz file save folder created ===")
    except OSError:
         print("=== FocusHawkEye npz file save folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    # train set image 저장되는 경로
    focus_train_directory = focus_main_directory + '/train'
    try:
        if not os.path.exists(focus_train_directory):
            os.makedirs(focus_train_directory)  # 디렉토리 생성 / train 파일 저장 폴더 생성
            print("=== FocusHawkEye train set save folder created ===")
    except OSError:
        print("=== FocusHawkEye train set save folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    # train set image npz 저장 경로
    focus_train_npz_directory = focus_main_directory + '/data'
    try:
        if not os.path.exists(focus_train_npz_directory):
            os.makedirs(focus_train_npz_directory)  # 디렉토리 생성 / train 파일 저장 폴더 생성
            print("=== FocusHawkEye train set npz file save folder created ===")
    except OSError:
        print("=== FocusHawkEye train set npz file save folder already exists ===")  # 이미 생성된 폴더의 경우 다음으로 넘어간다

    return app






# model = tf.keras.models.load_model("G:/내 드라이브/capstone_2/data/facenet_keras.h5")
# eye_model = load_model('2021_05_19_05_31_31.h5')
# ypr_model = load_model('model.h5')
# face_model = pickle.load(open('G:/내 드라이브/capstone_2/finalized_model.h5', 'rb'))



# 폴더명 정리
# 메인 = C:/FocusHawkEyeMain
# 웹캡 캡쳐 테스트 이미지 = C:/FocusHawkEyeMain/webCamCapture/temp
# npz 파일 저장 = C:/FocusHawkEyeMain/npzSave










