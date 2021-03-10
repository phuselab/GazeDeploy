#!/bin/bash

if [ -z "$1" ]
  then
    echo "No video file supplied"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No reference name supplied"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No output directory supplied"
    exit 1
fi

videofile=$1
ref_name=$2
out_dir=$3

V_feats_dir="${out_dir}pywork/${ref_name}/V_feats.npy"
A_feats_dir="${out_dir}pywork/${ref_name}/A_feats.npy"
out_scores="${out_dir}pywork/${ref_name}/"

tracks_dir="${out_dir}pywork/${ref_name}/tracks.pckl"
scores_dir="${out_scores}whospeaks.npy"
out_video="${out_dir}pyavi/${ref_name}/"
face_maps="speaker_detect/data/face_maps/"
face_maps_dir="speaker_detect/data/face_maps/${ref_name}/"
#STS_dir="speaker_detect/data/ST_maps/${ref_name}/"

sh speaker_detect/download_model.sh
python speaker_detect/run_pipeline_fixed.py --videofile $videofile --reference $ref_name --data_dir $out_dir
python speaker_detect/run_syncnet_fixed.py --videofile $videofile --reference $ref_name --data_dir $out_dir
python speaker_detect/get_speaker_score.py --video_feats $V_feats_dir --audio_feats $A_feats_dir --out_scores $out_scores

mkdir $face_maps
mkdir $face_maps_dir
#mkdir $STS_dir
python create_face_maps.py --tracks $tracks_dir --video $videofile --scores $scores_dir --out_video $out_video --maps_out $face_maps_dir
