# On gaze deployment to audio-visual cues of social interactions

***Giuseppe Boccignone, Vittorio Cuculo¹, Alessandro D'Amelio¹, Giuliano Grossi¹, Raffaella Lanzarotti¹***  
¹ [PHuSe Lab](https://phuselab.di.unimi.it) - Dipartimento di Informatica, Università degli Studi di Milano  

**Paper** TBA

![simulation](simulation.gif "Model Simulation")

### Requirements

```
pip install -r requirements.txt
```

### Executing the demo

To simulate from the model:

1. Do speaker identification and build face maps:
```
#sh build_face_maps.sh path/to/video vidName path/to/output
sh build_face_maps.sh data/videos/012.mp4 012 speaker_detect/output/
```
2. Run the follwing command (it is assumed that low-level saliency maps (see Credits) are already computed, if you want to compute it on your own, you may want to use something like [this](https://users.soe.ucsc.edu/~milanfar/research/rokaf/.html/SaliencyDetection.html#Matlab))
```
python3 Simulate_model.py
```

### Credits

- Data: FindWhoToLookAt --->[Repo](https://github.com/yufanLiu/find), [Paper](https://ieeexplore.ieee.org/document/8360155)

- Speaker Detection Pipeline: SyncNet ---> [Repo](https://github.com/joonson/syncnet_python), [Paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)

- Space-time Visual Saliency Detection: [Code](https://users.soe.ucsc.edu/~milanfar/research/rokaf/.html/SaliencyDetection.html#Matlab), [Paper](http://jov.arvojournals.org/article.aspx?articleid=2122209)

### Reference

If you use this code or data, please cite the paper:
```
TBA
```