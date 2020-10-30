import sys, os
import numpy as np
from shennong.audio import Audio
from shennong.features.processor.bottleneck import BottleneckProcessor

# Usage python src/extract_bnf.py sws2013Database_dev_eval/dev_queries/sws2013_dev_001.wav data/sws2013/dev_queries
wav_path = sys.argv[1]
out_dir  = sys.argv[2]

audio = Audio.load(wav_path)
processor = BottleneckProcessor(weights='BabelMulti')
features = processor.process(audio)

np.save(os.path.join(out_dir, os.path.basename(wav_path) + ".npy"), features.data)
