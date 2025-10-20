from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

import folium
import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import LineString, Point

import networkx as nx

from sklearn.neighbors import BallTree
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class LocalityGraph:
    """
    Create and manage a spatial graph from locality dictionaries.

    Typical locality dict keys used:
        - 'id' / 'nombre' / 'latitud_dec' / 'longitud_dec' / 'altitud' / 'num_hab'

    Example:
        lg = LocalityGraph.from_list(list_of_dicts)
        lg.build_graph(max_km=50)
        folium_map = lg.folium_map()
        folium_map.save("localities_graph.html")
    """

    def __init__(self, gdf: gpd.GeoDataFrame, logger: logging.Logger) -> None:
        """
        Initialize LocalityGraph with a GeoDataFrame (crs EPSG:4326 expected).

        Parameters
        ----------
        gdf:
            GeoDataFrame with columns 'id', 'nombre', 'lat', 'lon' and geometry points.
            Coordinates must be in decimal degrees and CRS EPSG:4326.
        logger:
            Logger instance for debug/info messages.
        """
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            raise ValueError("gdf must have CRS EPSG:4326 (WGS84)")
        if "id" not in gdf.columns:
            raise ValueError("gdf must contain an 'id' column")
        self.gdf = gdf.copy()
        # ensure lat/lon float columns exist
        self.gdf["lat"] = self.gdf.geometry.y.astype(float)
        self.gdf["lon"] = self.gdf.geometry.x.astype(float)
        self.gdf["altitud"] = self.gdf.get("altitud", pd.NA).astype(float)
        self.graph: Optional[nx.Graph] = None
        self.edges_gdf: Optional[gpd.GeoDataFrame] = None
        self.logger = logger
        self.logger.debug("LocalityGraph created with %d nodes", len(self.gdf))

    # -------------------------
    # Factory
    # -------------------------
    def parse_coordinate(value: Any) -> Optional[float]:
        """
        Parse a coordinate in several formats and return decimal degrees (float).
        Supported formats:
          - Decimal: "40.54845854" or 40.54845854
          - DMS with separators: 40º32'54.450744" or 40 32 54.450744
          - Compact DMS with hemisphere: "394924N" => 39°49'24" N
          - Signed DMS: "-0º48'28.084212\"" -> negative
          - Strings may contain whitespace
        Returns float degrees or None if cannot parse.
        """
        if value is None:
            return None
    
        s = str(value).strip()
        if s == "":
            return None
    
        # 1) If it's already a simple decimal number
        try:
            return float(s)
        except Exception:
            pass
        
        # 2) DMS with separators (º,°, ', ", spaces, colons)
        # Example: 40º32'54.450744"  or  -0 48 28.084212
        dms_re = re.compile(r"""
            ^\s*
            (?P<sign>[-+])?                  # optional sign
            \s*
            (?P<deg>\d{1,3})                 # degrees (1-3 digits)
            [^\d\w\-+]+                      # separator (º, space, :)
            (?P<min>\d{1,2})                 # minutes
            [^\d\w\-+]+                      # separator
            (?P<sec>\d{1,2}(?:\.\d+)?)       # seconds (with optional decimals)
            \s*
            (?P<hem>[NnSsEeWw])?             # optional hemisphere
            \s*$
        """, re.VERBOSE)
        m = dms_re.match(s)
        if m:
            deg = float(m.group("deg"))
            minute = float(m.group("min"))
            sec = float(m.group("sec"))
            sign = -1.0 if m.group("sign") == "-" else 1.0
            hem = m.group("hem")
            if hem:
                hem = hem.upper()
                if hem in ("S", "W"):
                    sign = -1.0
            return sign * (abs(deg) + minute / 60.0 + sec / 3600.0)
    
        # 3) Compact DMS like 394924N or 025309E or 003530S
        # Interpret as DDMMSSH or DDDMMSSH (if degrees could be 3 digits)
        compact_re = re.compile(r"^\s*(?P<num>\d{5,7})(?P<hem>[NnSsEeWw])\s*$")
        m2 = compact_re.match(s)
        if m2:
            num = m2.group("num")
            hem = m2.group("hem").upper()
            # decide split: if len 6 or 7, last two are seconds, previous two minutes, rest degrees
            # e.g. 394924 -> 39 49 24 ; 025309 -> 02 53 09 ; 1234567 -> 123 45 67
            n = len(num)
            sec = float(num[-2:])
            minute = float(num[-4:-2])
            deg = float(num[: n - 4])
            sign = -1.0 if hem in ("S", "W") else 1.0
            return sign * (deg + minute / 60.0 + sec / 3600.0)
    
        # 4) Fallback: try to extract groups of numbers (loose parsing)
        loose = re.findall(r"[-+]?\d+\.?\d*", s)
        if len(loose) >= 3:
            try:
                deg = float(loose[0])
                minute = float(loose[1])
                sec = float(loose[2])
                # Check for hemisphere letter at end
                hem = s[-1].upper() if s[-1].isalpha() else None
                sign = -1.0 if str(deg).startswith("-") else 1.0
                if hem in ("S", "W"):
                    sign = -1.0
                return sign * (abs(deg) + minute / 60.0 + sec / 3600.0)
            except Exception:
                pass
            
        # give up
        return None

    @classmethod
    def from_list(cls, localities: Iterable[Dict[str, Any]], logger: logging.Logger) -> "LocalityGraph":
        """
        Build LocalityGraph from an iterable of locality dicts.

        The function will try to read decimal coords from 'latitud_dec' / 'longitud_dec'
        and fall back to 'lat' / 'lon' if present.

        Returns
        -------
        LocalityGraph
        """
        rows: List[Dict[str, Any]] = []
        for loc in localities:
            # robust extraction with several possible keys
            lat = (
                loc.get("latitud_dec")
                or loc.get("lat")
                or loc.get("latitude")
                or loc.get("latitud")
            )
            lon = (
                loc.get("longitud_dec")
                or loc.get("lon")
                or loc.get("longitude")
                or loc.get("longitud")
            )
            lat_f = cls.parse_coordinate(value=lat)
            lon_f = cls.parse_coordinate(value=lon)
            if lat_f is None or lon_f is None:
                print("Skipping locality with unparsable coords: %r", loc)
                continue

            alt_raw = loc.get("altitud") or loc.get("alt") or loc.get("elevation")
            try:
                alt = float(alt_raw) if alt_raw is not None and alt_raw != "" else float("nan")
            except Exception:
                alt = float("nan")

            row = {
                "id": loc.get("id") or loc.get("id_old") or str(len(rows)),
                "nombre": loc.get("nombre") or loc.get("capital") or "",
                "lat": lat_f,
                "lon": lon_f,
                "altitud": alt,
                "num_hab": _safe_int(loc.get("num_hab")),
                "raw": loc,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"], df["altitud"]),
            crs="EPSG:4326",
        )
        return cls(gdf, logger)

    # -------------------------
    # Graph building
    # -------------------------
    def build_graph(self, max_km: float = 50.0, use_balltree: bool = True) -> None:
        """
        Build an undirected graph connecting nodes within max_km kilometers.

        Parameters
        ----------
        max_km:
            Maximum great-circle distance in kilometers to connect two nodes.
        use_balltree:
            If True and sklearn is available, use BallTree for O(n log n)
            neighbor lookup. Otherwise use an O(n^2) loop (fine for small n).
        """
        if use_balltree:
            self.logger.debug("Building graph with BallTree (sklearn) with max_km=%.2f", max_km)
            self.graph = self._build_with_balltree(max_km)
        else:
            if use_balltree:
                self.logger.info("scikit-learn not available; falling back to naive O(n^2) search")
            self.graph = self._build_naive(max_km)

        # build edges GeoDataFrame after creating graph
        self.edges_gdf = self._edges_to_gdf(self.graph)
        self.logger.info("Graph built: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())

    def _build_naive(self, max_km: float) -> nx.Graph:
        """
        Naive O(n^2) neighbor search based on geodesic approximation (haversine).
        Suitable for small datasets (< ~2000 nodes).
        """
        G = nx.Graph()
        nodes = list(self.gdf.itertuples(index=False))
        for r in nodes:
            G.add_node(r.id, nombre=r.nombre, lat=float(r.lat), lon=float(r.lon), altitud=r.altitud)

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1, alt1_m = nodes[i].lat, nodes[i].lon, nodes[i].altitud
                lat2, lon2, alt2_m = nodes[j].lat, nodes[j].lon, nodes[j].altitud
                dist_km = _distance_3d_m(lat1, lon1, alt1_m, lat2, lon2, alt2_m)
                if dist_km <= max_km * 1000.0:
                    G.add_edge(nodes[i].id, nodes[j].id, weight_km=dist_km)
        return G

    def _build_with_balltree(self, max_km: float) -> nx.Graph:
        """
        Use sklearn BallTree with haversine metric for efficient neighbor search.
        Coordinates are converted to radians for BallTree.
        """
        coords = np.vstack([self.gdf["lat"].values, self.gdf["lon"].values]).T
        # BallTree expects (lat, lon) in radians or (lon, lat)? We use radians in order (lat, lon)
        rad = np.deg2rad(coords)
        tree = BallTree(rad, metric="haversine")
        # query radius in radians
        radius = max_km / 6371.0088  # earth radius in km
        # query all neighbors within radius
        indices = tree.query_radius(rad, r=radius)

        G = nx.Graph()
        ids = list(self.gdf["id"].values)
        lat_vals = list(self.gdf["lat"].values)
        lon_vals = list(self.gdf["lon"].values)
        altitude_vals = list(self.gdf["altitud"].values)
        for idx, neighbors in enumerate(indices):
            node_id = ids[idx]
            # add node attrs
            G.add_node(node_id, nombre=self.gdf.iloc[idx]["nombre"], lat=lat_vals[idx], lon=lon_vals[idx], altitud=altitude_vals[idx])
            for nb in neighbors:
                if nb == idx:
                    continue
                other_id = ids[nb]
                # compute exact haversine distance (km)
                dist_km = _distance_3d_m(lat_vals[idx], lon_vals[idx], altitude_vals[idx], lat_vals[nb], lon_vals[nb], altitude_vals[nb])
                # add edge (undirected); avoid double-adding by comparing ids
                if node_id < other_id:
                    G.add_edge(node_id, other_id, weight_km=dist_km)
        return G

    # -------------------------
    # Exports / helpers
    # -------------------------
    @staticmethod
    def _edges_to_gdf(graph: nx.Graph) -> gpd.GeoDataFrame:
        """
        Convert NetworkX edges into a GeoDataFrame with LineString geometries.
        Expects node attributes 'lat' and 'lon' present.
        """
        rows: List[Dict[str, Any]] = []
        for u, v, data in graph.edges(data=True):
            uattr = graph.nodes[u]
            vattr = graph.nodes[v]
            geom = LineString([(uattr["lon"], uattr["lat"]), (vattr["lon"], vattr["lat"])])
            rows.append({"u": u, "v": v, "weight_km": data.get("weight_km"), "geometry": geom})
        egdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        return egdf

    def save_geojson(self, nodes_path: str = "nodes.geojson", edges_path: str = "edges.geojson") -> None:
        """
        Save nodes and edges as GeoJSON files.
        """
        self.logger.debug("Saving nodes to %s and edges to %s", nodes_path, edges_path)
        self.gdf.to_file(nodes_path, driver="GeoJSON")
        if self.edges_gdf is not None:
            self.edges_gdf.to_file(edges_path, driver="GeoJSON")

    # -------------------------
    # QC and plotting
    # -------------------------
    def geopandas_qc_plot(self) -> None:
        """
        Simple static plot using GeoPandas / Matplotlib:
            - nodes colored by degree
            - edges overplotted
        """
        if self.graph is None or self.edges_gdf is None:
            raise RuntimeError("Graph is not built. Call build_graph() first.")

        deg = dict(self.graph.degree())
        self.gdf["degree"] = self.gdf["id"].map(deg).fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(10, 10))
        base = self.gdf.plot(ax=ax, column="degree", cmap="plasma", legend=True, markersize=20)
        self.edges_gdf.plot(ax=base, linewidth=0.7, alpha=0.6)
        ax.set_title("Localities (colored by degree) and edges")
        plt.show()

    # -------------------------
    # Interactive folium map
    # -------------------------
    def folium_map(self, start_zoom: int = 6) -> "folium.Map":
        """
        Build a Folium interactive map with nodes and edges.
        Each node popup contains name, id and degree.
        Returns the Folium Map instance.
        """
        import folium  # local import

        if self.graph is None or self.edges_gdf is None:
            raise RuntimeError("Graph is not built. Call build_graph() first.")

        mean_lat = float(self.gdf["lat"].mean())
        mean_lon = float(self.gdf["lon"].mean())
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=start_zoom)

        # draw edges
        for row in self.edges_gdf.itertuples(index=False):
            coords = [(c[1], c[0]) for c in list(row.geometry.coords)]
            folium.PolyLine(coords, weight=1, color="blue", opacity=0.6).add_to(m)

        # draw nodes
        for r in self.gdf.itertuples(index=False):
            deg = 0
            try:
                deg = int(nx.degree(self.graph, r.id))
            except Exception:
                deg = 0
            popup = folium.Popup(f"{r.nombre}<br>id={r.id}<br>deg={deg}", max_width=250)
            folium.CircleMarker(location=(r.lat, r.lon), radius=4, popup=popup, color="red", fill=True).add_to(m)

        return m

    # -------------------------
    # Utilities
    # -------------------------
    def isolated_nodes(self) -> List[str]:
        """Return list of node ids that have degree 0."""
        if self.graph is None:
            raise RuntimeError("Graph is not built. Call build_graph() first.")
        return [n for n, d in self.graph.degree() if d == 0]

    def degree_stats(self) -> Dict[str, Any]:
        """Return simple degree stats: min, max, mean, median."""
        if self.graph is None:
            raise RuntimeError("Graph is not built. Call build_graph() first.")
        degs = np.array([d for _, d in self.graph.degree()])
        return {
            "min": int(degs.min()) if len(degs) else 0,
            "max": int(degs.max()) if len(degs) else 0,
            "mean": float(degs.mean()) if len(degs) else 0.0,
            "median": float(np.median(degs)) if len(degs) else 0.0,
        }

# -------------------------
# Helper functions
# -------------------------
def _safe_float(val: Optional[Any]) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _safe_int(val: Optional[Any]) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance between two points in decimal degrees.
    Returns kilometers.
    """
    # convert degrees to radians
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0088 * c  # Earth radius in km

def _distance_3d_m(lat1: float, lon1: float, alt1_m: float, lat2: float, lon2: float, alt2_m: float) -> float:
    """
    3D distance in meters: combine haversine horizontal (converted to meters)
    with vertical difference via Pythagoras.
    """
    horiz_km = _haversine_km(lat1, lon1, lat2, lon2)
    horiz_m = horiz_km * 1000.0
    dz = (alt2_m or 0.0) - (alt1_m or 0.0)
    return math.sqrt(horiz_m * horiz_m + dz * dz)
