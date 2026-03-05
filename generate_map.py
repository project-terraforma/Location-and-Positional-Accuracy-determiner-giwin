import pandas as pd
import folium

def create_visual_map():
    print("Loading data...")
    # Load the data that has calculating offsets
    df = pd.read_csv("ground_truth_with_errors.csv")
    
    # Sort by error size so we don't crowd the map with 0m offset pins.
    # Let's show the top 100 worst offenders so the story of "fixing errors" is clear.
    df = df.sort_values(by='offset_meters', ascending=False).head(100)
    
    # Center map roughly on the US (since most points are here)
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    print("Plotting Original vs True pins on map...")
    for idx, row in df.iterrows():
        # The bad, original overture location (RED)
        orig_popup = f"<b>{row['primary_name']}</b><br>Overture Original (Error: {row['offset_meters']:.1f}m)"
        folium.CircleMarker(
            location=[row['original_lat'], row['original_lon']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=orig_popup
        ).add_to(m)
        
        # The newly hand-labeled Ground Truth Location (GREEN)
        true_popup = f"<b>{row['primary_name']}</b><br>TruePin (Ground Truth)"
        folium.CircleMarker(
            location=[row['truth_lat'], row['truth_lon']],
            radius=5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
            popup=true_popup
        ).add_to(m)
        
        # Draw a line connecting the bad pin to the good pin (shows the distance moved)
        folium.PolyLine(
            locations=[
                [row['original_lat'], row['original_lon']], 
                [row['truth_lat'], row['truth_lon']]
            ],
            color='blue',
            weight=2,
            opacity=0.5
        ).add_to(m)
        
    out_file = "offset_map.html"
    m.save(out_file)
    print(f"Interactive map created! Open {out_file} in any web browser to view.")

if __name__ == "__main__":
    create_visual_map()
