import os
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth)

def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
        
    return tracks

def extract_features(tracks):
    track_ids = [t['track']['id'] for t in tracks if t['track']]
    features = sp.audio_features(track_ids)
    return pd.DataFrame(features)

# Example: 'Peaceful Piano' playlist (chill mood)
playlist_id = "37i9dQZF1DX4sWSpwq3LiO"
tracks = get_playlist_tracks(playlist_id)
df = extract_features(tracks)

# Keep only numerical features + track name
useful_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'liveness', 'speechiness']
df_clean = df[useful_cols].dropna()

# Pairplot
sns.pairplot(df_clean)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
