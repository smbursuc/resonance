<h2>Resonance</h2>

<div>
  Resonance is a tool for analyzing music that extracts music theory compliant and relevant features that help define the song's melody. It can be used in order to scan YouTube playlists and do feature extraction for each track found in the playlist. The end goal is mapping extracted features to musical and audible features such that it is easy to identify songs that are similar to each other
  based on the quantized musical feature. The tool can be used to analyze and find similar music.

  TODO: post the paper link for the project
</div>

<h2>Prerequisites</h2>

RMVPE is required to be imported. Get it here: https://github.com/Dream-High/RMVPE/tree/main<br>
Install libraries as needed (torch, librosa etc.)<br>
Note that for running the CREPE tests you might need to make another virtual environment, as RMVPE and CREPE use a different numpy version.

For this to work, you have to have the RMVPE repository in the parent folder of the resonance repo.

Example:<br>
`/path/to/resonance`<br>
`/path/to/RMVPE`<br>

<h2>How to run</h2>

1. Make sure RMVPE works first (you can do a separate testcase for RMVPE if needed).
2. Run `python resonance.py <your_youtube_playlist_link>` (make sure the playlist is public).
3. Wait for the execution to finish. The script generates a CSV file and several plot images with the results.

<h2>Tools</h2>

The analysis scripts can be run standalone, for example `python .\chroma\chroma.py`, if the clustering is not essential. There are multiple scripts that extract certain features or plot graphs.

<h2>Conclusion</h2>

Feel free to change threshold values (e.g `MIN_SUSTAINED_DURATION_SEC` in `clusterize.py` for sustained pitches) and experiment with the results!
