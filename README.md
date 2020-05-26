# On gaze deployment to audio-visual cues of social interactions

***Giuseppe Boccignone, Vittorio Cuculo¹, Alessandro D'Amelio¹, Giuliano Grossi¹, Raffaella Lanzarotti¹***  
¹ [PHuSe Lab](https://phuselab.di.unimi.it) - Dipartimento di Informatica, Università degli Studi di Milano  

**Paper** TBA

![simulation](simulation.gif "Model Simulation")

### Requirements

```
requirements.txt
```

### Executing the demo

To simulate from the model:

1. Do speaker identification and build face maps:
```
#sh build_face_maps.sh path/to/video vidName path/to/output
sh build_face_maps.sh data/videos/012.mp4 012 speaker_detect/output/
```
2. Run the follwing command (it is assumed that low-level saliency maps are already computed)
```
python3 Simulate_model.py

```

### Reference

If you use this code or data, please cite the paper:
```
TBA
```