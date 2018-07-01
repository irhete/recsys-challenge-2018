"""
-----------------------------------------------------------------
            
            RecSys Challenge 2018 - Team Latte


           _..,---,.._
        .-;'-.,___,.-';        Irene Teinemaa [irheta@ut.ee] 
       (( |           |        2018.07.01
        `  \         /
          _ `,.___.,'-
         (   '-----'   )
          -.._______..-
          
-----------------------------------------------------------------
""" 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
import time
import pickle
from collections import defaultdict

import pandas as pd

import metrics

start = time.time()

if len(sys.argv) != 2:
    print()
    print('----> Usage: python3 validate.py file_to_be_evaluated.csv')
    print()
    exit(1)

input_file = sys.argv[1]
print('- Evaluation file: {}'.format(input_file))

df = pd.read_csv(input_file, sep=',', skiprows=1, header=None)
data = df.to_dict('records')
pid_recs_map = {}

for d in data:
    pid = d[0]
    recs = [d[i] for i in range(1, 501)]
    pid_recs_map[pid] = recs

with open('track_to_data_all.pickle', 'rb') as handle:
    track_to_data = pickle.load(handle)
    
#Load validation set
with open('val_holdout_data_10000.pickle', 'rb') as handle:
    val_holdout_tracks = pickle.load(handle)

sum_rprec = 0
sum_ndcg = 0
sum_clicks = 0
sum_rprec_arti = 0
rprec_by_group = defaultdict(float)
ndcg_by_group = defaultdict(float)
clicks_by_group = defaultdict(float)
rprec_arti_by_group = defaultdict(float)
group_counts = defaultdict(int)

total_playlists = len(pid_recs_map)

for pid in pid_recs_map:
    
    recs = pid_recs_map[pid]
    holdout_tracks = [track['track_uri'] for track in val_holdout_tracks[pid]["tracks"]]

    rprec = metrics.r_precision(holdout_tracks, recs)
    ndcg = metrics.ndcg(holdout_tracks, recs, 500)
    clicks = metrics.playlist_extender_clicks(holdout_tracks, recs, 500)
    rprec_arti = metrics.r_precision_with_artist_fallback(holdout_tracks, recs, track_to_data)

    sum_rprec += rprec
    sum_ndcg += ndcg
    sum_clicks += clicks
    sum_rprec_arti += rprec_arti

    sample_type = val_holdout_tracks[pid]["sample_type"]
    rprec_by_group[sample_type] += rprec
    ndcg_by_group[sample_type] += ndcg
    clicks_by_group[sample_type] += clicks
    rprec_arti_by_group[sample_type] += rprec_arti
    group_counts[sample_type] += 1
    

for sample_type, count in group_counts.items():
    print(sample_type, "RPrec: ", rprec_by_group[sample_type] / count, ", NDCG: ", ndcg_by_group[sample_type] / count, ", Clicks: ", clicks_by_group[sample_type] / count, ", RPrec artist: ", rprec_arti_by_group[sample_type] / count)

print()
print("Overall RPrec: ", sum_rprec / total_playlists, ", NDCG: ", sum_ndcg / total_playlists, ", Clicks: ", sum_clicks / total_playlists, ", RPrec artist: ", sum_rprec_arti / total_playlists)
print()
