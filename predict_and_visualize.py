import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import h3
import overturemaps
import numpy as np
import xgboost as xgb
import folium
import os
import random

def get_buffered_bbox(lat, lon, buffer_meters=150):
    lat_buffer = buffer_meters / 111111.0
    lon_buffer = buffer_meters / (111111.0 * np.cos(np.radians(lat)))
    return (lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)

def main():
    print("1. Loading Data...")
    # Get the 500 IDs we already labeled so we don't accidentally predict on them
    labeled_df = pd.read_csv("ground_truth_fully_analyzed.csv")
    labeled_ids = set(labeled_df['id'].unique())
    
    # Load raw parquet
    parquet_path = "/Users/giwin/Documents/CRWN 102(New Attempt)/project_d_samples.parquet"
    raw_df = pd.read_parquet(parquet_path)
    
    import shapely.wkb as wkb
    raw_df['geometry'] = raw_df['geometry'].apply(lambda x: wkb.loads(x) if x is not None else None)
    
    # Filter out ones we already labeled
    unlabeled_df = raw_df[~raw_df['id'].isin(labeled_ids)].copy()
    
    # Grab 25 random locations to predict
    sample_df = unlabeled_df.sample(25, random_state=42).copy()
    
    # Needs a mock structural category for the model (in reality we would hit the LLM again)
    # We will just assign Standalone for simplicity of this visual demo
    sample_df['struct_cat'] = 'Standalone'
    
    print("2. Generating H3 Grids and Fetching Overture Features for Unseen Data...")
    enriched_hexes = []
    
    import time
    for idx, row in sample_df.iterrows():
        # Get point coordinates (WKT point to lat/lon)
        pt = row['geometry']
        if pt is None or pt.is_empty:
            continue
        lon, lat = pt.x, pt.y
        point_id = row['id']
        
        # 1. Generate K-Ring 2 (19 hexes)
        center_hex = h3.latlng_to_cell(lat, lon, 11)
        neighborhood = h3.grid_disk(center_hex, 2)
        
        local_hexes = []
        for hex_id in neighborhood:
            boundary = h3.cell_to_boundary(hex_id)
            shapely_poly = Polygon([(b_lon, b_lat) for b_lat, b_lon in boundary])
            local_hexes.append({
                'original_point_id': point_id,
                'h3_index': hex_id,
                'is_center': 1 if hex_id == center_hex else 0,
                'hex_geometry': shapely_poly,
                'original_lat': lat,
                'original_lon': lon
            })
            
        hex_gdf = gpd.GeoDataFrame(local_hexes, geometry='hex_geometry', crs="EPSG:4326")
        local_hexes_proj = hex_gdf.to_crs("EPSG:3857")
        center_hex_proj = local_hexes_proj[local_hexes_proj['is_center'] == 1].geometry.iloc[0].centroid
        
        # 2. Fetch Overture Features
        bbox = get_buffered_bbox(lat, lon, 150)
        try:
            buildings_gdf = overturemaps.core.dataframe("building", bbox=bbox)
            has_bldgs = len(buildings_gdf) > 0
            bldgs_proj = buildings_gdf.to_crs("EPSG:3857") if has_bldgs else None
        except:
            has_bldgs = False
            
        try:
            roads_gdf = overturemaps.core.dataframe("segment", bbox=bbox)
            has_roads = len(roads_gdf) > 0
            roads_proj = roads_gdf.to_crs("EPSG:3857") if has_roads else None
        except:
            has_roads = False
            
        # 3. Engineer row features
        for h_idx, h_row in local_hexes_proj.iterrows():
            hex_geom = h_row['hex_geometry']
            hex_centroid = hex_geom.centroid
            
            dist_from_center = hex_centroid.distance(center_hex_proj)
            
            intersects_building = False
            bldg_area = 0.0
            if has_bldgs:
                intersecting = bldgs_proj[bldgs_proj.intersects(hex_geom)]
                if not intersecting.empty:
                    intersects_building = True
                    bldg_area = intersecting.geometry.area.max()
                    
            dist_to_road = 999.0
            if has_roads:
                dist_to_road = roads_proj.distance(hex_centroid).min()
                
            enriched_hexes.append({
                'original_point_id': point_id,
                'h3_index': h_row['h3_index'],
                'is_center': h_row['is_center'],
                'dist_from_center': dist_from_center,
                'intersects_building': 1 if intersects_building else 0,
                'max_building_area': bldg_area,
                'dist_to_road': dist_to_road,
                'cat_Large Area': 0,      # Dummies
                'cat_Mall or Nested': 0,  # Dummies
                'cat_Standalone': 1,      # Assigned standalone for demo
                'cat_Skyscraper': 0,      # Dummies
                'hex_geometry': h_row['hex_geometry'], # Keep for mapping
                'original_lat': h_row['original_lat'],
                'original_lon': h_row['original_lon']
            })
            
        time.sleep(0.2)
        
    predict_df = pd.DataFrame(enriched_hexes)
    
    print("3. Training XGBoost Model on the 500 hand-labeled Ground Truth Points...")
    train_df = pd.read_csv("h3_classification_features.csv")
    
    # Prepare training features
    dummies = pd.get_dummies(train_df['struct_cat'], prefix='cat')
    train_df = pd.concat([train_df, dummies], axis=1)
    
    # Ensure all columns exist
    for col in ['cat_Large Area', 'cat_Mall or Nested', 'cat_Standalone', 'cat_Skyscraper']:
        if col not in train_df.columns:
            train_df[col] = 0
            
    feature_cols = ['is_center', 'dist_from_center', 'intersects_building', 'max_building_area', 'dist_to_road',
                    'cat_Large Area', 'cat_Mall or Nested', 'cat_Standalone', 'cat_Skyscraper']
                    
    X_train = train_df[feature_cols]
    y_train = train_df['is_true_pin']
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    print("4. Predicting New Locations...")
    X_pred = predict_df[feature_cols]
    probs = model.predict_proba(X_pred)[:, 1]
    predict_df['pred_prob'] = probs
    
    print("5. Generating Prediction Map visualization...")
    # Center map on US roughly
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    prediction_results = []
    
    for loc_id, group in predict_df.groupby('original_point_id'):
        # original overture
        orig_lat = group.iloc[0]['original_lat']
        orig_lon = group.iloc[0]['original_lon']
        
        # predicted hex
        best_hex_row = group.sort_values('pred_prob', ascending=False).iloc[0]
        pred_hex_id = best_hex_row['h3_index']
        
        # Get lat/lon of the predicted hex center
        pred_lat, pred_lon = h3.cell_to_latlng(pred_hex_id)
        
        place_name = sample_df[sample_df['id'] == loc_id].iloc[0]['names']
        if isinstance(place_name, dict) and 'primary' in place_name:
             name_str = place_name['primary']
        else:
             name_str = "Unknown Place"

        # Map Original (Red)
        orig_popup = f"<b>{name_str}</b><br>Overture Original Pin"
        folium.CircleMarker(
            location=[orig_lat, orig_lon],
            radius=6, color='red', fill=True, fill_color='red', fill_opacity=0.9, popup=orig_popup
        ).add_to(m)
        
        # Map Prediction (Blue)
        pred_popup = f"<b>{name_str}</b><br>✨ XGBoost Predicted Pin"
        folium.CircleMarker(
            location=[pred_lat, pred_lon],
            radius=6, color='blue', fill=True, fill_color='blue', fill_opacity=0.9, popup=pred_popup
        ).add_to(m)
        
        # Draw dotted line connecting them
        folium.PolyLine(
            locations=[[orig_lat, orig_lon], [pred_lat, pred_lon]],
            color='purple', weight=3, opacity=0.8, dash_array='5, 5'
        ).add_to(m)
        
    out_file = "model_prediction_map.html"
    m.save(out_file)
    print(f"\nSUCCESS! AI Prediction map created: {out_file}")
    print("The model successfully placed pins on unseen data! Open the HTML file to view.")

if __name__ == "__main__":
    main()
