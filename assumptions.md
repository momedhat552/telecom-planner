# Assumptions

> Status: early planning note, kept for history. The final study area is Berlin (see README.md);
> the Mohandessin / Dokki area was dropped because OpenStreetMap has too many missing buildings.

## Goal
Plan base station sites and a frequency reuse map for a city area, comparing
a greedy coverage algorithm against a plain hexagonal grid.

## Study area
Mohandessin + Dokki, Giza. About 3 x 3 km around (30.047 N, 31.205 E).
Exact bounding box is fixed in Phase 1.

## Radio assumptions (all to be verified)
- Band: 3.5 GHz (n78)
- Bandwidth: 100 MHz (typical n78 carrier; real operator allocation may be smaller)
- Base station antenna height: 25 m (3GPP UMa default)
- UE (phone) height: 1.5 m
- Transmit power: 46 dBm per sector (assumed, no antenna gain model in v1)
- Covered means RSRP >= -100 dBm (assumed threshold)

## Propagation model
3GPP TR 38.901 Urban Macro (UMa). Free-space rejected: it ignores clutter and
is too optimistic in dense urban areas.

## Demand model
Demand per grid square is proportional to building footprint area from OSM.
Simplification: ignores building height, use (home/office) and time of day.

## Baseline
Plain hexagonal grid using the same number of sites as the greedy result.

## Evaluation metrics
- % of area covered
- % of demand covered
- Number of sites
- Mean RSRP over covered squares

## Known limitations
- No terrain or building-height shadowing
- OSM may be incomplete or out of date
- No real traffic data
- Single band, no interference model beyond frequency colouring