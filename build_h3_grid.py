import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import h3
import numpy as np

def generate_h3_neighborhood(lat, lon, original_id, resolution=11, k_ring=1):
    """
    Given a point, generates the center H3 hexagon and its K-Ring neighbors.
    Returns a list of dictionaries with H3 indices and their geometries.
    """
    # 1. Get the center hex (h3 v4 API)
    center_hex = h3.latlng_to_cell(lat, lon, resolution)
    
    # 2. Get the neighborhood (k_ring=1 gives exactly 7 hexes. k_ring=2 gives 19).
    neighborhood = h3.grid_disk(center_hex, k_ring)
    
    hex_data = []
    for hex_id in neighborhood:
        # Get the polygon boundary for this hex (h3 v4 API)
        boundary = h3.cell_to_boundary(hex_id)
        # H3 returns (lat, lon). Shapely wants (lon, lat)
        shapely_poly = Polygon([(lon, lat) for lat, lon in boundary])
        
        hex_data.append({
            'original_point_id': original_id,
            'h3_index': hex_id,
            'is_center': hex_id == center_hex,
            'hex_geometry': shapely_poly
        })
        
    return hex_data


def main():
    print("Loading analyzed ground truth data...")
    # Read the data that has both the original Overture locations and our TruePin labels
    df = pd.read_csv("ground_truth_fully_analyzed.csv")
    
    # We want to use the ORIGINAL overture point to build the grid neighborhood
    # because in the real world, the model only has the original overture point to start with.
    
    K_RING_SIZE = 2 # 19 hexes per location
    print(f"Generating H3 Discretized Grids (Resolution 11, K-Ring {K_RING_SIZE}) for {len(df)} POIs...")
    
    all_hexes = []
    for idx, row in df.iterrows():
        neighborhood = generate_h3_neighborhood(
            row['original_lat'], 
            row['original_lon'], 
            row['id'], 
            resolution=11, 
            k_ring=K_RING_SIZE
        )
        all_hexes.extend(neighborhood)
        
    # Create a GeoDataFrame of all our discrete hex cells
    hex_gdf = gpd.GeoDataFrame(all_hexes, geometry='hex_geometry', crs="EPSG:4326")
    print(f"Total H3 Grid Cells generated: {len(hex_gdf)}")
    
    # Now, we need to label the TARGET CLASS for our ML model.
    # Which of these hexes actually contains the TruePin (Ground Truth)?
    print("Assigning TruePin targets to grid cells...")
    
    # Create a geodataframe of just the ground truth points
    truth_geom = [Point(xy) for xy in zip(df['truth_lon'], df['truth_lat'])]
    truth_gdf = gpd.GeoDataFrame(df[['id']], geometry=truth_geom, crs="EPSG:4326")
    
    # A hex cell gets a '1' if it contains the truth point for its parent ID, otherwise '0'
    hex_gdf['is_true_pin'] = 0
    captured_count = 0
    
    # Iterate through each unique original place
    for place_id in df['id'].unique():
        # Get the truth point for this place
        truth_pt = truth_gdf[truth_gdf['id'] == place_id]
        if truth_pt.empty:
            continue
            
        # Get the index of the hexes that belong to this place
        place_hex_indices = hex_gdf[hex_gdf['original_point_id'] == place_id].index
        
        # Check which hex the truth point intersects
        for hex_idx in place_hex_indices:
            if hex_gdf.loc[hex_idx, 'hex_geometry'].contains(truth_pt.iloc[0].geometry):
                hex_gdf.at[hex_idx, 'is_true_pin'] = 1
                captured_count += 1
                break # Only one hex can contain the point
                
    # See how many we captured 
    print(f"Success: The true pin is inside our local H3 grid for {captured_count} out of {len(df)} locations.")
    
    if captured_count < len(df) * 0.5:
         print(f"Warning: We might need a larger K-Ring (currently {K_RING_SIZE}) to capture those massive outliers!")
         
    # Save the basic grid structure before adding heavy Overture features
    hex_gdf.drop(columns=['hex_geometry']).to_csv("h3_classification_base.csv", index=False)
    print("Saved base grid structure to h3_classification_base.csv")
    
if __name__ == "__main__":
    main()
