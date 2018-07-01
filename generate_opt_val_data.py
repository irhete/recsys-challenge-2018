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

import json
import codecs
import pickle
import random
import pandas as pd


for data_type in ["opt", "val"]:

    if data_type == "opt":
        val_data_filename = 'opt_data_10000.json'
        val_holdout_data_filename = 'opt_holdout_data_10000.pickle'
        slice_start_idx = 980000
        slice_end_idx = 990000
        
    elif data_type == "val":
        val_data_filename = 'val_data_10000.json'
        val_holdout_data_filename = 'val_holdout_data_10000.pickle'
        slice_start_idx = 990000
        slice_end_idx = 1000000
    
    else:
        print("Data type unknown.")

    dt_ids_sizes = pd.DataFrame()
    for slice_start in range(slice_start_idx, slice_end_idx, 1000):
        path = "data/mpd.slice.%s-%s.json" % (slice_start, slice_start + 999)
        f = codecs.open(path, 'r', 'utf-8')
        js = f.read()
        f.close()
        playlists = json.loads(js)['playlists']
        dt_ids_sizes = pd.concat([dt_ids_sizes, pd.DataFrame({"ids": [playlist["pid"] for playlist in playlists],
                                                              "sizes": [len(playlist["tracks"]) for playlist in playlists]})], axis=0)


    val_sample_assignments = {}
    dt_to_choose_from = dt_ids_sizes.copy()
    group_size = int(len(dt_ids_sizes)/10)

    # 100 random tracks
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 100]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["100_random"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 100 first tracks
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 100]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["100_first"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 25 random tracks
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 25]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["25_random"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 25 first tracks
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 25]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["25_first"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 10 first tracks without title
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 10]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["10_no_title"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 10 first tracks with title
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 10]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["10_with_title"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 5 first tracks without title
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 5]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["5_no_title"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 5 first tracks with title
    tmp = dt_to_choose_from[dt_to_choose_from.sizes > 5]
    sample = tmp.sample(n=group_size, random_state=22).ids
    val_sample_assignments["5_with_title"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # 1 first track
    sample = dt_to_choose_from.sample(n=group_size, random_state=22).ids
    val_sample_assignments["1_with_title"] = list(sample)
    dt_to_choose_from = dt_to_choose_from[~dt_to_choose_from.ids.isin(sample)]

    # title only
    val_sample_assignments["title_only"] = list(dt_to_choose_from.ids)


    val_sample_assignments_by_pid = {pid: sample for sample, pids in val_sample_assignments.items() for pid in pids}


    val_playlists = {'playlists': []}
    val_holdout_tracks = {}
    random.seed(22)
    for slice_start in range(slice_start_idx, slice_end_idx, 1000):
        path = "data/mpd.slice.%s-%s.json" % (slice_start, slice_start + 999)
        f = codecs.open(path, 'r', 'utf-8')
        js = f.read()
        f.close()
        playlists = json.loads(js)['playlists']
        for playlist in playlists:
            val_playlist = {}
            if val_sample_assignments_by_pid[playlist["pid"]] == "title_only":
                val_playlist["tracks"] = []
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "1_with_title":
                val_playlist["tracks"] = playlist["tracks"][:1]
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "5_with_title":
                val_playlist["tracks"] = playlist["tracks"][:5]
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "5_no_title":
                val_playlist["tracks"] = playlist["tracks"][:5]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "10_with_title":
                val_playlist["tracks"] = playlist["tracks"][:10]
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "10_no_title":
                val_playlist["tracks"] = playlist["tracks"][:10]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "25_first":
                val_playlist["tracks"] = playlist["tracks"][:25]
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "25_random":
                random.shuffle(playlist["tracks"])
                val_playlist["name"] = playlist["name"]
                val_playlist["tracks"] = playlist["tracks"][:25]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "100_first":
                val_playlist["tracks"] = playlist["tracks"][:100]
                val_playlist["name"] = playlist["name"]

            elif val_sample_assignments_by_pid[playlist["pid"]] == "100_random":
                random.shuffle(playlist["tracks"])
                val_playlist["tracks"] = playlist["tracks"][:100]
                val_playlist["name"] = playlist["name"]

            val_playlist["pid"] = playlist["pid"]
            val_playlist["num_samples"] = len(val_playlist["tracks"])
            val_playlist["num_tracks"] = len(playlist["tracks"])
            val_playlist["num_holdouts"] = val_playlist["num_tracks"] - val_playlist["num_samples"]
            val_playlists['playlists'].append(val_playlist)

            val_holdout_tracks[playlist["pid"]] = {}
            val_holdout_tracks[playlist["pid"]]["tracks"] = [track for track in playlist["tracks"] if track not in val_playlist["tracks"]]
            val_holdout_tracks[playlist["pid"]]["sample_type"] = val_sample_assignments_by_pid[playlist["pid"]]


    with open(val_data_filename, 'w') as outfile:
        json.dump(val_playlists, outfile)

    with open(val_holdout_data_filename, 'wb') as handle:
        pickle.dump(val_holdout_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)
