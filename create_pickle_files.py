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

import time
import sys
import json
import codecs
import pickle
import string
from collections import Counter


start = time.time()

print("Generating intermediate structures ...")

data_types = ["opt", "val", "submission"]

for data_type in data_types:
    if data_type == "val":
        path = "val_data_10000.json"
    elif data_type == "submission":
        path = "challenge_set.json"
    elif data_type == "opt":
        path = "opt_data_10000.json"

    track_uris = set()
    artist_uris = set()
    album_uris = set()
    words = set()
    translator = str.maketrans('','',string.punctuation)
    f = codecs.open(path, 'r', 'utf-8')
    js = f.read()
    f.close()
    playlists = json.loads(js)['playlists']
    for playlist in playlists:
        for track in playlist['tracks']:
            track_uris.add(track['track_uri'])
            artist_uris.add(track['artist_uri'])
            album_uris.add(track['album_uri'])

        if "name" in playlist:
            for word in playlist['name'].split():
                words.add(word.translate(translator).lower())
                
    with open('%s_track_uris.pickle' % data_type, 'wb') as handle:
        pickle.dump(track_uris, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s_artist_uris.pickle' % data_type, 'wb') as handle:
        pickle.dump(artist_uris, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s_album_uris.pickle' % data_type, 'wb') as handle:
        pickle.dump(album_uris, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s_words.pickle' % data_type, 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

playlist_idx = 0
track_to_data = {}

print("Generating track_to_data map ..." )
track_uri_counter = Counter()
for slice_start in range(0, 1000000, 1000):
    path = "data/mpd.slice.%s-%s.json" % (slice_start, slice_start + 999)
    f = codecs.open(path, 'r', 'utf-8')
    js = f.read()
    f.close()
    playlists = json.loads(js)['playlists']
    for playlist in playlists:
        for track in playlist['tracks']:
            track_to_data[track['track_uri']] = {}
            track_to_data[track['track_uri']]['album_uri'] = track['album_uri']
            track_to_data[track['track_uri']]['artist_uri'] = track['artist_uri']
            track_uri_counter[track['track_uri']] += 1
            
        playlist_idx += 1
        if playlist_idx % 100000 == 0:
            print("%s playlists, %s seconds" % (playlist_idx, time.time() - start))
            
with open('track_to_data_all.pickle', 'wb') as handle:
    pickle.dump(track_to_data, handle)

n_top_songs = len(track_uri_counter)
top_songs_counts = dict(track_uri_counter.most_common(n_top_songs))
with open('top_%s_songs.pickle' % n_top_songs, 'wb') as handle:
    pickle.dump(top_songs_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
n_top_songs = 100000
top_songs_counts = dict(track_uri_counter.most_common(n_top_songs))
with open('top_%s_songs.pickle' % n_top_songs, 'wb') as handle:
    pickle.dump(top_songs_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Done: %s seconds" % (time.time() - start))
