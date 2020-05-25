import numpy as np
import pickle as pkl
from skimage.draw import polygon, polygon_perimeter
import skvideo.io
from skimage.transform import resize
import argparse
import ntpath
from skimage import measure
import skimage
import matplotlib.pyplot as plt
import os
import imageio

def get_face_blob(img_shape, x, y, s, k1=0.6, k2=0.7):

	img = np.zeros(img_shape)
	r_center = y
	c_center = x
	r_radius = s * k2
	c_radius = s * k1

	rr, cc = skimage.draw.ellipse(r=r_center, c=c_center, r_radius=r_radius, c_radius=c_radius, rotation=0)
	rr[rr>=img_shape[0]] = img_shape[0]-1
	cc[cc>=img_shape[1]] = img_shape[1]-1
	img = np.zeros(img_shape, dtype=np.uint8)
	img[rr, cc] = 1

	return img

'''

This script takes as input from the command line:

--tracks: Path to tracks files, produced by run_pipeline.py - usually in the pywork directory
--video: Path to original video
--scores: Path to the speaker scores file produced by get_speaker_score.py
--out_file: Path and filename of the output video i.e. the original video plus speaker and non speaker bboxes
--bboxes_out: Output path for saving bounding boxes, a file containing the bounding boxes for each frame and for each subject of the video

The output of the script is a video, written to the --out_file directory. Red bboxes represent the current speaker, blue one(s) non speaker(s)

'''

parser = argparse.ArgumentParser(description = "Get_speaker_score")
parser.add_argument('--tracks', type=str, default='', help='Path to tracks files, produced by run_pipeline.py')
parser.add_argument('--video', type=str, default='', help='Path to original video')
parser.add_argument('--scores', type=str, default='', help='Path to the speaker scores file produced by get_speaker_score.py')
parser.add_argument('--out_video', type=str, default='', help='Path and filename of the output video')
parser.add_argument('--maps_out', type=str, default='', help='Output path and filename for saving face maps')
opt = parser.parse_args()


with open(opt.tracks, 'rb') as fil:
	tracks = pkl.load(fil)


videodata = skvideo.io.vread(opt.video)
whospeaks = np.load(opt.scores)
vdata_boxes = []
vidName = os.path.basename(opt.video)[:-4]

nfaces = len(tracks)
nframes = min(len(tracks[0][0][0]), len(whospeaks), videodata.shape[0])

face_boxes = {}
for face in range(nfaces):
	face_boxes[str(face)] = []

for frame in range(nframes):

	print('Processing frame ' + str(frame+1) + ' of ' + str(nframes))

	whospeaks_now = whospeaks[frame]
	curr_frame = videodata[frame,:,:,:]
	img_shape = curr_frame.shape[:2]

	speaker_map = np.zeros(img_shape)
	non_speaker_map = np.zeros(img_shape)

	for face in range(nfaces):
		
		sizes = tracks[face][1][0]
		xs = tracks[face][1][1]
		ys = tracks[face][1][2]
	
		s = int(sizes[frame])
		x = int(xs[frame])
		y = int(ys[frame])
		s = int(s*1.8)

		start = np.array([x-s, y-s])
		extent = np.array(np.array([x+s, y+s] - start))

		m = get_face_blob(img_shape, x, y, s)

		p1 = (start[0], start[1])
		p2 = (start[0]+extent[0], start[1])
		p3 = (start[0]+extent[0], start[1]+extent[1])
		p4 = (start[0], start[1]+extent[1])

		face_boxes[str(face)].append([p1,p2,p3,p4])

		#for skimage polygon
		r = np.array([p1[0], p2[0], p3[0], p4[0]])
		c = np.array([p1[1], p2[1], p3[1], p4[1]])

		#rr, cc = rectangle(start, extent=extent, shape=img_size)
		rr, cc = polygon_perimeter(r, c)
		#img[rr, cc] = 1

		if face == whospeaks_now:
			curr_frame[cc-1,rr-1,:] = np.array([255,0,0])
			curr_frame[cc,rr,:] = np.array([255,0,0])
			curr_frame[cc+1,rr+1,:] = np.array([255,0,0])
			speaker_map += m
		else:
			curr_frame[cc-1,rr-1,:] = np.array([0,0,255])
			curr_frame[cc,rr,:] = np.array([0,0,255])
			curr_frame[cc+1,rr+1,:] = np.array([0,0,255])
			non_speaker_map += m

	vdata_boxes.append(np.expand_dims(curr_frame, axis=0))

	#save maps for the current frame
	name_speaker = opt.maps_out + vidName + '_f' + str(frame) + '_speaker.png'
	name_non_speaker = opt.maps_out + vidName + '_f' + str(frame) + '_nonspeaker.png'

	imageio.imwrite(name_speaker, speaker_map.astype('uint8')*255)
	imageio.imwrite(name_non_speaker, non_speaker_map.astype('uint8')*255)

boxed_video = np.vstack(vdata_boxes)

print(boxed_video.shape)
print(videodata.shape)

video_fname = ntpath.basename(opt.video)

v_split = video_fname.split('.')
video_fname = v_split[0]
extension = v_split[1]

writer = skvideo.io.FFmpegWriter(opt.out_video + video_fname + '_boxed.' + extension, outputdict={'-b': '300000000'})
for i in range(boxed_video.shape[0]):
	writer.writeFrame(boxed_video[i, :, :, :])
	
writer.close()

#np.save(opt.bboxes_out + 'bboxes.npy', face_boxes)

