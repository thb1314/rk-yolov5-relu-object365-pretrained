# Official YOLOv5 is the consumer interface

V1 documents official Ultralytics YOLOv5 v6.2 `train.py` as the supported fine-tuning
entry point. MMYOLO produced the source checkpoints and remains available through
training records, but exposing it as a second user workflow would add dependencies and
make the first release less approachable without improving downstream RKNN deployment.
