import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import overturemaps
import numpy as np
import os
import time

def get_buffered_bbox(lat, lon, buffer_meters=150):
    """
    Returns a bounding box (xmin, ymin, xmax, ymax) centered on lat,lon
    buffered by approximately the specified meters.
    """
    # Rough approximation: 1 degree latitude is ~111,111 meters
    lat_buffer = buffer_meters / 111111.0
    # Longitude shrinks as we move toward poles
    lon_buffer = buffer_meters / (111111.0 * np.cos(np.radians(lat)))
    
    xmin = lon - lon_buffer
    ymin = lat - lat_buffer
    xmax = lon + lon_buffer
    ymax = lat + lat_buffer
    
    return (xmin, ymin, xmax, ymax)

def engineer_features():
    print("Loading base H3 grid (h3_classification_base.csv)...")
    hex_df = pd.read_csv("h3_classification_base.csv")
    
    # Needs geometry back for spatial operations
    import h3
    from shapely.geometry import Polygon
    
    def hex_to_poly(hex_id):
        boundary = h3.cell_to_boundary(hex_id)
        return Polygon([(lon, lat) for lat, lon in boundary])
        
    print("Reconstructing hex geometries...")
    hex_df['hex_geometry'] = hex_df['h3_index'].apply(hex_to_poly)
    hex_gdf = gpd.GeoDataFrame(hex_df, geometry='hex_geometry', crs="EPSG:4326")
    
    print("Loading point data for bounding boxes...")
    points_df = pd.read_csv("ground_truth_fully_analyzed.csv")
    
    total_locations = len(points_df)
    print(f"Starting Overture spatial feature fetch for {total_locations} locations...")
    
    # We will accumulate the enriched hexes here
    enriched_hexes = []
    
    # File to save progress to in case of failure
    temp_save_file = "temp_hex_features.csv"
    start_index = 0
    
    if os.path.exists(temp_save_file):
        existing_df = pd.read_csv(temp_save_file)
        # Find which location IDs we already processed
        processed_ids = set(existing_df['original_point_id'].unique())
        print(f"Found {len(processed_ids)} already processed locations. Resuming...")
        enriched_hexes = existing_df.to_dict('records')
    else:
        processed_ids = set()

    # Track how many API calls fail
    failed_calls = 0

    for idx, row in points_df.iterrows():
        point_id = row['id']
        
        if point_id in processed_ids:
            continue
            
        print(f"Processing {idx+1}/{total_locations}: {row['primary_name']}...")
        
        # Get bounding box ~150m around the point to cover the K-Ring=2 grid completely
        bbox = get_buffered_bbox(row['original_lat'], row['original_lon'], 150)
        
        # Pull Overture Buildings for this bbox
        try:
            buildings_gdf = overturemaps.core.dataframe("building", bbox=bbox)
            has_buildings = len(buildings_gdf) > 0
        except Exception as e:
            # If nothing exists or error
            buildings_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
            has_buildings = False

        # Pull Overture Roads (segment)
        try:
            roads_gdf = overturemaps.core.dataframe("segment", bbox=bbox)
            has_roads = len(roads_gdf) > 0
        except Exception as e:
            roads_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
            has_roads = False

        # Get the 19 hexes for this specific point
        local_hexes = hex_gdf[hex_gdf['original_point_id'] == point_id].copy()
        
        # Convert Geometries to a projected coordinate system (meters) for accurate distance math
        # 3857 is Web Mercator (meters)
        local_hexes_proj = local_hexes.to_crs("EPSG:3857")
        buildings_proj = buildings_gdf.to_crs("EPSG:3857") if has_buildings else buildings_gdf
        roads_proj = roads_gdf.to_crs("EPSG:3857") if has_roads else roads_gdf
        
        center_hex_proj = local_hexes_proj[local_hexes_proj['is_center'] == True].geometry.iloc[0].centroid
        
        # Calculate features for each hex
        new_hex_data = []
        for h_idx, h_row in local_hexes_proj.iterrows():
            hex_geom = h_row['hex_geometry']
            hex_centroid = hex_geom.centroid
            
            # 1. Distance from the original center overture point
            dist_from_center = hex_centroid.distance(center_hex_proj)
            
            # 2. Building Intersection & Area
            intersects_building = False
            bldg_area = 0.0
            if has_buildings:
                intersecting_bldgs = buildings_proj[buildings_proj.intersects(hex_geom)]
                if not intersecting_bldgs.empty:
                    intersects_building = True
                    # Take the area of the largest intersecting building
                    bldg_area = intersecting_bldgs.geometry.area.max()
            
            # 3. Distance to nearest road
            dist_to_road = 999.0
            if has_roads:
                dist_to_road = roads_proj.distance(hex_centroid).min()
                
            # Pack it up - grab the original unprojected row data
            h_dict = local_hexes.loc[h_idx].to_dict()
            h_dict['dist_from_center'] = dist_from_center
            h_dict['intersects_building'] = 1 if intersects_building else 0
            h_dict['max_building_area'] = bldg_area
            h_dict['dist_to_road'] = dist_to_road
            
            # Add context features from parent point
            h_dict['struct_cat'] = row['structural_category']
            
            # Drop the geometry object so it saves to CSV cleanly
            del h_dict['hex_geometry']
            new_hex_data.append(h_dict)
            
        enriched_hexes.extend(new_hex_data)
        processed_ids.add(point_id)
        
        # Save progress every 10 locations
        if len(processed_ids) % 10 == 0:
            pd.DataFrame(enriched_hexes).to_csv(temp_save_file, index=False)
            
        # Optional: Tiny sleep to not spam AWS/Overture endpoints
        time.sleep(0.5)
        
    # Final save
    final_df = pd.DataFrame(enriched_hexes)
    final_df.to_csv("h3_classification_features.csv", index=False)
    print("\nSUCCESS! Saved final ML feature dataset to: h3_classification_features.csv")
    
    # Cleanup temp
    if os.path.exists(temp_save_file):
        os.remove(temp_save_file)
        
if __name__ == "__main__":
    engineer_features()
