import os
import argparse
from time import time
import logging as log
import cv2
import numpy as np
import json
import requests
import queue
import threading
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from collections import defaultdict


tic = time()
auth = HTTPBasicAuth()
users = {
    "a": generate_password_hash("0202")
}

def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def pprint(mydict):
	print(json.dumps(mydict, sort_keys=True, indent=4))

class VideoCapture2:
	def __init__(self, name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()

parser = argparse.ArgumentParser()
parser.add_argument(
	"-c", "--confidence", type=float, default=0.05,	help="confidence threshold"
)
parser.add_argument(
	"-nms", "--non_maximum", type=float, default=0.5, help="non-maximum supression threshold"
)
parser.add_argument(
	"-ll", "--logging_level", type=str, default=30, help="logging level"
)
parser.add_argument(
	"-r", "--render", type=bool, default=False, help="return with base64 image and rendering"
)
args = vars(parser.parse_args())
log.basicConfig(level=args["logging_level"])
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
classes = list()
with open("resources/labels.txt", "r") as f:
	classes = [cname.strip() for cname in f.readlines()]
NN = cv2.dnn.readNetFromDarknet(
	"resources/yolov4-tiny.cfg",
	"resources/yolov4-tiny.weights"
)
NN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
NN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
NN_layer = NN.getLayerNames()
NN_layer = [NN_layer[i - 1] for i in NN.getUnconnectedOutLayers()]
cv2.cuda.setDevice(0)
allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
app = Flask(__name__)
log.info("initialized [{:.4f} ms]".format(time()-tic))
log.info("ready")

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/config', methods=['GET', 'POST'])
@auth.login_required
def config():
	if request.method == 'POST':
		data = request.json
		if not (data.get('confidence') is None):
			args["confidence"] = data.get('confidence')
			log.debug("confidence: {}".format(args["confidence"]))
		if not (data.get('non_maximum') is None):
			args["non_maximum"] = data.get('non_maximum')
			log.debug("non_maximum: {}".format(args["non_maximum"]))
		if not (data.get('logging_level') is None):
			args["logging_level"] = data.get('logging_level')
			log.getLogger().setLevel(data.get('logging_level'))
			log.debug("logging_level: {}".format(args["logging_level"]))
		if not (data.get('render') is None):
			args["render"] = data.get('render')
			log.debug("render: {}".format(args["render"]))
		return "configured!", 200
	if request.method == 'GET':
		return '''
			<a href="/" alt="home">home</a> / <a href="/config" alt="home">config</a> <br />
			<h1>POST configurations</h1>
			<h3> available parameters: </h3>
			confidence [{}] <br />
			non_maximum [{}] <br />
			logging_level [{}] <br />
			render [{}]
		'''.format(
			args["confidence"],
			args["non_maximum"],
			args["logging_level"],
			args["render"]
			), 200

@app.route("/",methods=['GET', 'POST'])
@auth.login_required
def index():
	if request.method == 'POST':
		if 'file' not in request.files:
			print('No file attached in request')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			print('No file selected')
			return redirect(request.url)
		if file and check_allowed_file(file.filename):
			payload = defaultdict(list)
			boxes = []
			classIds = []
			confidences = []
			tic = time()
			tic_origin = tic
			filename = secure_filename(file.filename)
			img = Image.open(file.stream)
			if img.mode != 'RGB':
				img = img.convert('RGB')
			open_cv_image = np.array(img) 
			frame = open_cv_image[:, :, ::-1].copy() 
			W, H = img.size
			payload["stats"].append(["pre-process (ms)", round((time()-tic)*1000)])
			tic = time()
			NN.setInput(
				cv2.dnn.blobFromImage(
					frame, 
					1/255.0, (416, 416),
					swapRB=True, 
					crop=False
				)
			)
			for output in NN.forward(NN_layer):
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > args["confidence"]:
						box = detection[0:4] * np.array([W, H, W, H])
						(center_x, center_y, width, height) = box.astype("int")
						x = int(center_x - (width/2))
						y = int(center_y - (height/2))
						boxes.append([x, y, int(width), int(height)])
						classIds.append(classID)
						confidences.append(float(confidence))
			payload["stats"].append(["detect (ms)", round((time()-tic)*1000)])
			tic = time()
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["non_maximum"])
			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					obj = ["{}".format(classes[classIds[i]]), [x,y,w,h], round(confidences[i], 4)]
					payload["detected"].append(obj)
					cv2.putText(frame, "{}".format(classes[classIds[i]]), (x+(int)(w/2), y+(int)(h/2)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			retval, buffer = cv2.imencode('.jpg', frame)
			encoded_string = base64.b64encode(buffer).decode() 
			toc = time()
			payload["stats"].append(["post-process (ms)", round((toc-tic)*1000)])
			payload["stats"].append(["total (ms)", round((toc-tic_origin)*1000)])
		response = json.dumps({
			"detected": payload["detected"],
			"stats": payload["stats"]
			})
		if args["render"]:
			return render_template('index.html', img_data=encoded_string, json_data=response), 200
		else:
			return response, 200
	else:
		return render_template('index.html', img_data="", json_data=""), 200

if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', port=5002)