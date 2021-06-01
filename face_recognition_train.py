# Modeling
import os
import numpy as np
#import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, utils
from tensorflow.keras import optimizers
from sklearn.cluster import KMeans
from keras.models import load_model
import mtcnn
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle

model = load_model("facenet_keras.h5")


# 주어진 사진에서 하나의 얼굴 추출
def extract_face(filename, required_size=(160, 160)):
	# 파일에서 이미지 불러오기
	image = Image.open(filename)
	# RGB로 변환, 필요시
	image = image.convert('RGB')
	# 배열로 변환
	pixels = asarray(image)
	# 감지기 생성, 기본 가중치 이용
	detector = MTCNN()
	# 이미지에서 얼굴 감지
	results = detector.detect_faces(pixels)
	#print(results)
 # 첫 번째 얼굴에서 경계 상자 추출
	x1, y1, width, height = results[0]['box']
	# 버그 수정
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# 얼굴 추출
	face = pixels[y1:y2, x1:x2]
  # 모델 사이즈로 픽셀 재조정
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# 디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출
def load_faces(directory):
	faces = list()
	# 파일 열거
	for filename in listdir(directory):
		# 경로
		path = directory + filename
		# 얼굴 추출
		face = extract_face(path)
		# 저장
		faces.append(face)
	return faces

# 이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기
def load_dataset(directory):
	X, y = list(), list()
	# 클래스별로 폴더 열거
	for subdir in listdir(directory):
		# 경로
		path = directory + subdir + '/'
		# 디렉토리에 있을 수 있는 파일을 건너뛰기(디렉토리가 아닌 파일)
		if not isdir(path):
			continue
		# 하위 디렉토리의 모든 얼굴 불러오기
		faces = load_faces(path)
		# 레이블 생성
		labels = [subdir for _ in range(len(faces))]
		# 진행 상황 요약
		print('>%d개의 예제를 불러왔습니다. 클래스명: %s' % (len(faces), subdir))
		# 저장
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)


# 하나의 얼굴의 얼굴 임베딩 얻기
def get_embedding(model, face_pixels):
	# 픽셀 값의 척도
	face_pixels = face_pixels.astype('int32')
	# 채널 간 픽셀값 표준화(전역에 걸쳐)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# 얼굴을 하나의 샘플로 변환
	samples = expand_dims(face_pixels, axis=0)
	# 임베딩을 갖기 위한 예측 생성
	yhat = model.predict(samples)
	return yhat[0]

def face_recognition_training():
	# 훈련 데이터셋 불러오기 -----> 여기 경로 수정해야한다. train 폴더 안에 user 이메일 별로 여러 폴더가 존재할 것임
	trainX, trainy = load_dataset('C:/FocusHawkEyeMain/train/')

	# 배열을 단일 압축 포맷 파일로 저장
	savez_compressed('C:/FocusHawkEyeMain/data/5-celebrity-faces-dataset_train.npz', trainX, trainy)

	# 얼굴 데이터셋 불러오기
	data_train = load('C:/FocusHawkEyeMain/data/5-celebrity-faces-dataset_train.npz')
	trainX, trainy = data_train['arr_0'], data_train['arr_1']

	# facenet 모델 불러오기
	model = load_model("facenet_keras.h5")

	# 훈련 셋에서 각 얼굴을 임베딩으로 변환하기
	newTrainX = list()
	for face_pixels in trainX:
		embedding = get_embedding(model, face_pixels)
		newTrainX.append(embedding)
	newTrainX = asarray(newTrainX)
	# print(newTrainX.shape)

	# 배열을 하나의 압축 포맷 파일로 저장
	savez_compressed('C:/FocusHawkEyeMain/data/5-celebrity-faces-embeddings_train.npz', newTrainX, trainy)

	# 데이터셋 불러오기
	data_train = load('C:/FocusHawkEyeMain/data/5-celebrity-faces-embeddings_train.npz')

	trainX, trainy = data_train['arr_0'], data_train['arr_1']

	# 입력 벡터 일반화
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)

	model = SVC(kernel='linear', probability=True)
	model.fit(trainX, trainy)
	# save the model to disk
	#filename = 'finalized_model_new.h5'
	filename = 'finalized_model.h5'      # 기존 모델 새로 트레인 후 업데이트된다 / face recognition 모델은 그룹마다 나눌 필요x
	pickle.dump(model, open(filename, 'wb'))

	print(str('+++ new model created +++'))

