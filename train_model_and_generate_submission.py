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
import os
import json
import codecs
import csv
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


data_type = argv[1] # val or submission
playlist_count = int(argv[2]) # 990000 for val, or 1000000 for submission
n_top_songs = 2262292 # all tracks

results_dir = "result_files"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

item_weight = {"title_only": 0,
               "1_with_title": 0.9035899566365081,
               "5_no_title": 0.710443592369379,
               "5_with_title": 0.9030390981835763,
               "10_no_title": 0.24861615664634532,
               "10_with_title": 0.5460159056402332,
               "25_first": 0.7462605324311976,
               "25_random": 0.970795096597976,
               "100_first": 0.832655382476601,
               "100_random": 0.9032361882321588}

word_weight = {"title_only": 1,
               "1_with_title": 0.3823669625845644,
               "5_no_title": 0,
               "5_with_title": 0.30436376565383694,
               "10_no_title": 0,
               "10_with_title": 0.5261905393594316,
               "25_first": 0.5929553709031262,
               "25_random": 0.42414578193470465,
               "100_first": 0.6890961507865375,
               "100_random": 0.9876776760475077}

albu_weight = {"title_only": 0,
               "1_with_title": 0.0006651329820028251,
               "5_no_title": 0.02808472023996831,
               "5_with_title": 0.005285773732074783,
               "10_no_title": 0.0008275872456813826,
               "10_with_title": 0.0013414442728668578,
               "25_first": 0.0008155601325202377,
               "25_random": 0.1407010673811464,
               "100_first": 0.023032512308417213,
               "100_random": 0.16728418535161227}

arti_weight = {"title_only": 0,
               "1_with_title": 0.009699798418418126,
               "5_no_title": 0.028388209287863016,
               "5_with_title": 0.009043259398499946,
               "10_no_title": 0.0021869127404520426,
               "10_with_title": 0.0005451291816757755,
               "25_first": 0.027342375228137948,
               "25_random": 0.02137285804953583,
               "100_first": 0.03721716274982824,
               "100_random": 0.13118265302146326}

song_to_album_ratio = {"title_only": 1000,
               "1_with_title": 1,
               "5_no_title": 2,
               "5_with_title": 2,
               "10_no_title": 2,
               "10_with_title": 2,
               "25_first": 2,
               "25_random": 2,
               "100_first": 2,
               "100_random": 3}

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
overall_top_songs = [song for song, count in sorted(all_track_counts.items(), key=itemgetter(1), reverse=True)][:600]
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

if data_type == "val":
    slice_end_idx = 990000
else:
    slice_end_idx = 1000000

for slice_start in range(0, slice_end_idx, 1000):
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


if data_type == "val":
    input_filename = "val_data_10000.json"
    output_file_name = os.path.join(results_dir, "validation.csv")
    
else:
    input_filename = "challenge_set.json"
    output_file_name = os.path.join(results_dir, "submission.csv")
    
# Read the challenge playlists
print("Reading the challenge playlists...")
start = time.time()
with codecs.open(input_filename, 'r', 'utf-8') as f:
    js = f.read()
playlists = json.loads(js)['playlists']

user_item_challenge = lil_matrix( (len(playlists), len(track_to_idx)), dtype=bool )
user_albu_challenge = lil_matrix( (len(playlists), len(album_to_idx)), dtype=int)
user_word_challenge = lil_matrix( (len(playlists), len(word_to_idx)), dtype=int )
user_arti_challenge = lil_matrix( (len(playlists), len(artist_to_idx)), dtype=int)
playlist_idx = 0
pid_to_sample_group = {}
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
            
    if len(playlist['tracks']) == 0:
        playlist_sample_group = "title_only"
    elif len(playlist['tracks']) == 1:
        playlist_sample_group = "1_with_title"
    elif len(playlist['tracks']) == 5 and "name" in playlist:
        playlist_sample_group = "5_with_title"
    elif len(playlist['tracks']) == 5 and "name" not in playlist:
        playlist_sample_group = "5_no_title"
    elif len(playlist['tracks']) == 10 and "name" in playlist:
        playlist_sample_group = "10_with_title"
    elif len(playlist['tracks']) == 10 and "name" not in playlist:
        playlist_sample_group = "10_no_title"
    elif len(playlist['tracks']) == 25 and max([tr["pos"] for tr in playlist['tracks']]) > 25:
        playlist_sample_group = "25_random"
    elif len(playlist['tracks']) == 25:
        playlist_sample_group = "25_first"
    elif len(playlist['tracks']) == 100 and max([tr["pos"] for tr in playlist['tracks']]) > 100:
        playlist_sample_group = "100_random"
    elif len(playlist['tracks']) == 100:
        playlist_sample_group = "100_first"
    else:
        print("unknown group", len(playlist['tracks']), [tr["pos"] for tr in playlist['tracks']])
        
    pid_to_sample_group[playlist["pid"]] = playlist_sample_group

    playlist_idx += 1
print("Done: %s seconds" % (time.time() - start))

print("Multiplying the matrices...")
start = time.time()
# Create recommendations for the challenge playlists
ibcf_recs = user_item_challenge.dot(item_item)
word_recs = user_word_challenge.dot(word_item)
albu_recs = user_albu_challenge.dot(albu_item)
arti_recs = user_arti_challenge.dot(arti_item)
print("Done: %s seconds" % (time.time() - start))

print("Recommending...")
start = time.time()

with open(output_file_name, "w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    spamwriter.writerow(["team_info","main","Latte","irheta@ut.ee"])    
    for idx, playlist in enumerate(playlists):
        playlist_sample_type = pid_to_sample_group[playlist["pid"]]
        scores = (item_weight[playlist_sample_type] * ibcf_recs.getrow(idx).toarray()[0] +
                  word_weight[playlist_sample_type] * word_recs.getrow(idx).toarray()[0] +
                  albu_weight[playlist_sample_type] * albu_recs.getrow(idx).toarray()[0] +
                  arti_weight[playlist_sample_type] * arti_recs.getrow(idx).toarray()[0])
       
        recommended_songs_scores = [(idx_to_track_all[k], score) for k, score in enumerate(scores) if score > 0]
        recommended_songs_scores = sorted(recommended_songs_scores, key=itemgetter(1), reverse=True)
        existing_songs = {track['track_uri']:1 for track in playlist["tracks"]}
        
        existing_albums = set([track_to_data[track]["album_uri"] for track in existing_songs.keys()])
        if len(existing_albums) > 0 and len(existing_songs) / len(existing_albums) > song_to_album_ratio[playlist_sample_type]:
            if "random" not in playlist_sample_type:
                recs = [song for song, score in recommended_songs_scores if song not in existing_songs and track_to_data[song]["album_uri"] == track_to_data[playlist["tracks"][-1]["track_uri"]]["album_uri"]][:500]
            else:
                recs = [song for song, score in recommended_songs_scores if song not in existing_songs and track_to_data[song]["album_uri"] in existing_albums][:500]
            
            if len(recs) < 500:
                recs += [song for song, score in recommended_songs_scores[:600] if song not in existing_songs and song not in recs][:500-len(recs)]
            
        else:
            recs = [song for song, score in recommended_songs_scores[:600] if song not in existing_songs][:500]
            
        # if too few rec-s, pad with overall popular songs
        if len(recs) < 500:
            recs += [song for song in overall_top_songs if song not in existing_songs and song not in recs][:500-len(recs)]
        spamwriter.writerow([playlist['pid']] + recs)
print("Done: %s seconds" % (time.time() - start))
