# Apex Analysis Tool

Currently implemented features:
- Localisation of player on a reference map based on minimap

Mid-term planned features:
- Basic player stats (kills, assists, damage) monitoring using OCR
- Player state monitoring based on CV
- Game state monitoring based on CV
- Advanced player stats monitoring (knocks, times knocked, times respawned, time spent in loot boxes)

Long-term planned features:
- Deep learning on extracted stats for player improvement

Localisation algorithm:
1) Crop current frame to contain only minimap
2) Compute SIFT features for reference map and minimap
3) Match features from reference map to minimap
4) Using the distance ratio to determine the best matches
5) Compute the scale difference between minimap and reference map by cross calculating all distances between the selected matched features
6) Using the scale difference, estimate the ego position by projecting the minimap centre on the reference map
7) Linear Kalman Filter to smooth position estimate 
