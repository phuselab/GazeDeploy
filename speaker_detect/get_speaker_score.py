import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import entropy_estimators as ee
from scipy.signal import medfilt
from scipy import stats
import argparse

'''
This script takes as input:

--video_feats: Path to SyncNet Video Features, the V_feats file produced by run_syncnet_fixed.py
--audio_feats: Path to SyncNet Audio Features, the A_feats file produced by run_syncnet_fixed.py
--whospeaks: Path for the speaker scores output file
--display: either 1 or 0, depending if you want to plot results or not

The script produces as output a file (--whospeaks) containing for each video frame an integer number corresponding to the face track of the speaker
'''

parser = argparse.ArgumentParser(description = "Get_speaker_score")
parser.add_argument('--video_feats', type=str, default='', help='Path to SyncNet Video Features')
parser.add_argument('--audio_feats', type=str, default='', help='Path to SyncNet Audio Features')
parser.add_argument('--out_scores', type=str, default='', help='Path for the scores output file')
parser.add_argument('--display', type=int, default=0, help='0 or 1 - display plots flag')
opt = parser.parse_args()

def smooth(a,WSZ):
	# a: NumPy 1-D array containing the data to be smoothed
	# WSZ: smoothing window size needs, which must be odd number,
	# as in the original MATLAB implementation
	out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
	r = np.arange(1,WSZ-1,2)
	start = np.cumsum(a[:WSZ-1])[::2]/r
	stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
	return np.concatenate((  start , out0, stop  ))

def mode_filter(signal, win_len):
	sig_len = signal.shape[0]
	filtered = np.zeros(sig_len)
	half_win = int(win_len//2.)

	for i in range(half_win, sig_len - half_win):
		win_values = signal[i-half_win : i+half_win+1]
		filtered[i] = stats.mode(win_values)[0]

	return filtered

A = np.load(opt.audio_feats, allow_pickle=True)
V = np.load(opt.video_feats, allow_pickle=True)

'''A_list = [a[np.newaxis,:,:] for a in A]
V_list = [v[np.newaxis,:,:] for v in V]
A = np.vstack(A_list)
V = np.vstack(V_list)
import code
code.interact(local=locals())'''


display_fig = opt.display

n_faces = V.shape[0]
n_frames = V.shape[1]
fs = 25.
win_len = 9
eps = np.finfo(np.float32).eps

dist = np.zeros(n_frames)
mi = np.zeros(n_frames)

if display_fig:
	fig2 = plt.figure()
	ax2 = fig2.subplots()

mis = []

t = np.linspace(0,n_frames/fs, n_frames)
for face in range(n_faces):

	print('\nProcessing Face number ' + str(face+1) + ' of ' + str(n_faces) + '...')

	for i in range(n_frames-win_len):
		currV = V[face,i:i+win_len,:]
		currA = A[face,i:i+win_len,:]

		mi[i+win_len] = np.abs(ee.mi(currV,currA))

	mi[:win_len] = np.ones(win_len)*mi[win_len]
	mi_smoothed = np.expand_dims(smooth(medfilt(mi,49), 9), axis=1)
	mis.append(mi_smoothed)
	if display_fig:
		ax2.plot(t, mi_smoothed, label='Face'+str(face))

if display_fig:
	ax2.legend()
	ax2.grid()
	ax2.set_title('Mutual Information between Audio and Video')

mis = np.hstack(mis)
whospeaks = np.zeros(mis.shape[0])


for i in range(mis.shape[0]):
	curr_val = mis[i,:]
	whospeaks[i] = np.argmax(curr_val)

mode_win_len = 17
whospeaks = mode_filter(whospeaks, mode_win_len)
whospeaks[:mode_win_len] = np.ones(mode_win_len)*whospeaks[mode_win_len]
whospeaks[-mode_win_len:] = np.ones(mode_win_len)*whospeaks[-mode_win_len]

if display_fig:
	plt.figure()
	plt.plot(t, whospeaks, 'r')
	plt.yticks(np.arange(0, n_faces, step=1))
	plt.xlabel('time')
	plt.ylabel('Face (track) number')
	plt.grid()
	plt.show()

np.save(opt.out_scores + 'whospeaks.npy', whospeaks)