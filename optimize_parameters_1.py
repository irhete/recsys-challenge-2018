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
import json
import codecs
import pickle
from operator import itemgetter
from sys import argv
import time
import metrics
import string

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt

from collections import defaultdict


playlist_count = int(argv[1]) # 400000
n_top_songs = int(argv[2]) # 100000
n_opt_iter = int(argv[3]) # 100
sample_group = argv[4] # one of: ["title_only", "1_with_title", "5_no_title", "5_with_title", "10_no_title", "10_with_title", "25_first", "25_random", "100_first", "100_random"]

data_type = "opt"
input_filename = 'opt_data_10000.json'
val_holdout_data_filename = 'opt_holdout_data_10000.pickle'

translator = str.maketrans('','',string.punctuation)

with open('top_%s_songs.pickle' % n_top_songs, 'rb') as handle:
    all_track_counts = pickle.load(handle)
with open('%s_track_uris.pickle' % data_type, 'rb') as handle:
    track_uris = list(pickle.load(handle))
with open('%s_artist_uris.pickle' % data_type, 'rb') as handle:
    artist_uris = list(pickle.load(handle))
with open('%s_album_uris.pickle' % data_type, 'rb') as handle:
    album_uris = list(pickle.load(handle))
with open('%s_words.pickle' % data_type, 'rb') as handle:
    words = list(pickle.load(handle))
    
with open('track_to_data_all.pickle', 'rb') as handle:
    track_to_data = pickle.load(handle)

track_to_idx_all = {song: idx for idx, song in enumerate(all_track_counts.keys())}
idx_to_track_all = {idx: song for song, idx in track_to_idx_all.items()}
overall_top_songs = [song for song, count in sorted(all_track_counts.items(), key=itemgetter(1), reverse=True)][:10000]
del all_track_counts

track_uris = [track for track in track_uris if track in track_to_idx_all]
track_to_idx = {track: idx for idx, track in enumerate(track_uris)}
idx_to_track = {idx: song for song, idx in track_to_idx.items()}
artist_to_idx = {artist: idx for idx, artist in enumerate(artist_uris)}
album_to_idx = {album: idx for idx, album in enumerate(album_uris)}
word_to_idx = {word: idx for idx, word in enumerate(words)}

# Create user-item matrix from a sample of playlists
print("Reading the MPD playlists to a user-item matrix...")
start = time.time()
playlist_idx = 0
user_item = lil_matrix((playlist_count,len(track_to_idx_all)), dtype=float)
word_item = lil_matrix((len(word_to_idx), len(track_to_idx_all)), dtype=int)
albu_item = lil_matrix( (len(album_to_idx), len(track_to_idx_all)), dtype=int)
arti_item = lil_matrix( (len(artist_to_idx), len(track_to_idx_all)), dtype=int)

for slice_start in range(0, 980000, 1000):
    path = "data/mpd.slice.%s-%s.json" % (slice_start, slice_start + 999)
    f = codecs.open(path, 'r', 'utf-8')
    js = f.read()
    f.close()
    playlists = json.loads(js)['playlists']
    for playlist in playlists:
        for track in playlist['tracks']:
            if track["track_uri"] in track_to_idx_all:
                user_item[playlist_idx,track_to_idx_all[track['track_uri']]] = True

                if track['artist_uri'] in artist_to_idx:
                    arti_item[artist_to_idx[track['artist_uri']],track_to_idx_all[track['track_uri']]] += 1
                if track['album_uri'] in album_to_idx:
                    albu_item[album_to_idx[track['album_uri']],track_to_idx_all[track['track_uri']]] +=1
                for word in playlist['name'].split():
                    if word.translate(translator).lower() in word_to_idx:
                        word_item[word_to_idx[word.translate(translator).lower()],track_to_idx_all[track['track_uri']]] += 1

        playlist_idx += 1
        if playlist_idx % 100000 == 0:
            print("%s playlists, %s seconds" % (playlist_idx, time.time() - start))
            
        if playlist_idx >= playlist_count:
            break
    if playlist_idx >= playlist_count:
        break
print("Done: %s seconds" % (time.time() - start))

# Normalize the "ratings" for each user and calculate item-item similarities
print("Calculating the item-item similarities...")
start = time.time()
idxs_of_top_songs = [track_to_idx_all[idx_to_track[idx]] for idx in range(len(idx_to_track))]
item_item = user_item.tocsc()[:,idxs_of_top_songs].tolil().T.dot(user_item)
del user_item
word_item = normalize(word_item, norm='l1', axis=1)
albu_item = normalize(albu_item, norm='l1', axis=1)
arti_item = normalize(arti_item, norm='l1', axis=1)
item_item = normalize(item_item, norm='l1', axis=1)
print("Done: %s seconds" % (time.time() - start))


# Read the challenge playlists
print("Reading the challenge playlists...")
start = time.time()
with codecs.open(input_filename, 'r', 'utf-8') as f:
    js = f.read()
playlists = json.loads(js)['playlists']
with open(val_holdout_data_filename, 'rb') as handle:
    val_holdout_tracks = pickle.load(handle)
playlists = [pl for pl in playlists if val_holdout_tracks[pl["pid"]]["sample_type"] == sample_group]
    
user_item_challenge = lil_matrix( (len(playlists), len(track_to_idx)), dtype=bool )
user_albu_challenge = lil_matrix( (len(playlists), len(album_to_idx)), dtype=int)
user_word_challenge = lil_matrix( (len(playlists), len(word_to_idx)), dtype=int )
user_arti_challenge = lil_matrix( (len(playlists), len(artist_to_idx)), dtype=int)
playlist_idx = 0
for playlist in playlists:
    if 'name' in playlist:
        for word in playlist['name'].split():
            if word.translate(translator).lower() in word_to_idx:
                user_word_challenge[playlist_idx,word_to_idx[word.translate(translator).lower()]] += 1
    for track in playlist['tracks']:
        if track['track_uri'] in track_to_idx:
            user_item_challenge[playlist_idx,track_to_idx[track['track_uri']]] = True
        if track['artist_uri'] in artist_to_idx:
            user_arti_challenge[playlist_idx,artist_to_idx[track['artist_uri']]] += 1
        if track['album_uri'] in album_to_idx:
            user_albu_challenge[playlist_idx,album_to_idx[track['album_uri']]] += 1

    playlist_idx += 1
print("Done: %s seconds" % (time.time() - start))

print("Multiplying the matrices...")
start = time.time()
# Create recommendations for the challenge playlists
user_ibcf_recs = user_item_challenge.dot(item_item)
user_word_recs = user_word_challenge.dot(word_item)
user_albu_recs = user_albu_challenge.dot(albu_item)
user_arti_recs = user_arti_challenge.dot(arti_item)
print("Done: %s seconds" % (time.time() - start))

def evaluate_ndcg(args):
    
    item_weight = args.get("item_weight", 0)
    word_weight = args.get("word_weight", 0)
    albu_weight = args.get("albu_weight", 0)
    arti_weight = args.get("arti_weight", 0)
    song_to_album_ratio = args.get("song_to_album_ratio", 500)
    
    # Create recommendations for the challenge playlists
    user_item_recs = (item_weight * user_ibcf_recs + 
                      word_weight * user_word_recs + 
                      albu_weight * user_albu_recs + 
                      arti_weight * user_arti_recs)

    start = time.time()

    sum_ndcg = 0
    for idx, playlist in enumerate(playlists):
        recommended_songs_scores = [(idx_to_track_all[k], score) for k, score in enumerate(user_item_recs.getrow(idx).toarray()[0]) if score > 0]
        existing_songs = {track['track_uri']:1 for track in playlists[idx]["tracks"]}
        existing_albums = set([track_to_data[track]["album_uri"] for track in existing_songs.keys()])
        existing_artists = defaultdict(int)
        
        if "random" not in sample_group and len(existing_albums) > 0 and len(existing_songs) / len(existing_albums) > song_to_album_ratio:
            potential_recs = [track for track, vals in track_to_data.items() if vals["album_uri"] == track_to_data[playlist["tracks"][-1]["track_uri"]]["album_uri"]]
            
        else:
            potential_recs = []
            
        potential_recs += [song for song, score in sorted(recommended_songs_scores, key=itemgetter(1), reverse=True)]
        potential_recs += overall_top_songs
        
        recs = []
        for song in potential_recs:
            if len(recs) == 500:
                break
            if song not in existing_songs and song not in recs and existing_artists[track_to_data[song]["artist_uri"]] < args['max_songs_by_same_artist']:
                recs.append(song)
                existing_artists[track_to_data[song]["artist_uri"]] += 1
            
        
        holdout_tracks = [track['track_uri'] for track in val_holdout_tracks[playlist["pid"]]["tracks"]]
        
        sum_ndcg += metrics.ndcg(holdout_tracks, recs, 500)
    return {'loss': -sum_ndcg / len(playlists), 'status': STATUS_OK, 'model': None}


if sample_group == "title_only":
    space = {'word_weight': hp.uniform("word_weight", 0, 1),
             'max_songs_by_same_artist': hp.uniform("max_songs_by_same_artist", 1, 500)}
elif "no_title" in sample_group:
    space = {'albu_weight': hp.uniform("albu_weight", 0, 1),
             'arti_weight': hp.uniform("arti_weight", 0, 1),
             'item_weight': hp.uniform("item_weight", 0, 1),
             'song_to_album_ratio': hp.uniform("song_to_album_ratio", 0, 100),
             'max_songs_by_same_artist': hp.uniform("max_songs_by_same_artist", 1, 500)}
elif "random" in sample_group:
    space = {'albu_weight': hp.uniform("albu_weight", 0, 1),
             'arti_weight': hp.uniform("arti_weight", 0, 1),
             'item_weight': hp.uniform("item_weight", 0, 1),
             'word_weight': hp.uniform("word_weight", 0, 1),
             'max_songs_by_same_artist': hp.uniform("max_songs_by_same_artist", 1, 500)}
else:
    space = {'albu_weight': hp.uniform("albu_weight", 0, 1),
             'arti_weight': hp.uniform("arti_weight", 0, 1),
             'item_weight': hp.uniform("item_weight", 0, 1),
             'word_weight': hp.uniform("word_weight", 0, 1),
             'song_to_album_ratio': hp.uniform("song_to_album_ratio", 0, 100),
             'max_songs_by_same_artist': hp.uniform("max_songs_by_same_artist", 1, 500)}

print("Optimizing the weights...")
start = time.time()
trials = Trials()
print("Done: %s seconds" % (time.time() - start))

best = fmin(evaluate_ndcg, space, algo=tpe.suggest, max_evals=n_opt_iter, trials=trials)

best_params = hyperopt.space_eval(space, best)
print(best_params)