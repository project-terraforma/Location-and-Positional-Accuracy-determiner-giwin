import streamlit as st
import pandas as pd
import json
import os
import folium
from streamlit_folium import st_folium

# --- Setup and Load Data ---
st.set_page_config(page_title="Ground Truth Labeling App", layout="wide")

@st.cache_data
def load_data():
    with open("ground_truth_sample_1000.geojson", "r") as f:
        data = json.load(f)
    return data['features']

features = load_data()
total_points = len(features)

# --- CSV for Saving Ground Truth ---
OUTPUT_CSV = "ground_truth_labels.csv"

def load_labels():
    if os.path.exists(OUTPUT_CSV):
        return pd.read_csv(OUTPUT_CSV)
    else:
        return pd.DataFrame(columns=["id", "primary_name", "primary_category", "alternate_categories", "source_datasets", "original_lon", "original_lat", "truth_lat", "truth_lon", "notes"])

labels_df = load_labels()

# Find the first completely unlabeled point
unlabeled_indices = [i for i, f in enumerate(features) if f['properties']['id'] not in labels_df['id'].values]

if 'current_index' not in st.session_state:
    st.session_state.current_index = unlabeled_indices[0] if unlabeled_indices else 0

current_feature = features[st.session_state.current_index]
props = current_feature['properties']
coords = current_feature['geometry']['coordinates']
orig_lon, orig_lat = coords[0], coords[1]


# --- Sidebar: Criteria and Navigation ---
with st.sidebar:
    st.title("🗺️ Ground Truth Labeling")
    st.progress(len(labels_df) / total_points if total_points > 0 else 0)
    st.write(f"**Labeled:** {len(labels_df)} / {total_points}")
    
    st.markdown("---")
    st.markdown("### 📋 Labeling Criteria")
    st.info("""
    **Where does the pin go?**
    - 🏪 **Standalone Store:** Main pedestrian entrance.
    - 🛍️ **Stores inside a Mall/Strip Mall:** The specific exterior entrance closest to that store if it has one. If deep inside an enclosed mall with no exterior door, place the pin at the **main mall entrance** closest to where a user would park.
    - 🌲 **Large Area (Park, RV):** Main access road entrance/check-in.
    - 🏢 **Skyscraper:** Primary street-level lobby entrance.
    - 🛒 **Nested (Starbucks in Target):** Main entrance of the *parent* building.
    """)
    st.markdown("---")
    
    if st.button("⏪ Previous"):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()
            
    if st.button("⏭️ Skip / Next"):
        if st.session_state.current_index < total_points - 1:
            st.session_state.current_index += 1
            st.rerun()

# --- Main Window: Data & Maps ---
st.header(f"Point {st.session_state.current_index + 1}: {props.get('primary_name', 'Unknown')}")
st.write(f"**Primary Category:** {props.get('primary_category', 'Unknown')} | **Brand:** {props.get('brand_name', 'None')} | **Confidence:** {props.get('confidence', 'N/A')}")
st.write(f"**Alternate Categories:** {props.get('alternate_categories', 'None')}")
st.write(f"**Sources:** {props.get('source_datasets', 'Unknown')}")

col1, col2 = st.columns([1, 1])

# Column 1: Folium Map (Interactive placement)
with col1:
    st.subheader("1. Click Map to set Ground Truth")
    
    # We use a Google Satellite tile layer
    tiles = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    attr = "Google Satellite"
    
    m = folium.Map(location=[orig_lat, orig_lon], zoom_start=20, tiles=tiles, attr=attr)
    
    # Add Google Maps hybrid (streets + satellite) as an option
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Hybrid',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Marker for Overture's original location
    folium.Marker(
        [orig_lat, orig_lon], 
        popup="Overture Original Location", 
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Add click functionality
    st_data = st_folium(m, width=700, height=500)
    
    truth_lat, truth_lon = None, None
    if st_data and st_data.get("last_clicked"):
        truth_lat = st_data["last_clicked"]["lat"]
        truth_lon = st_data["last_clicked"]["lng"]
        st.success(f"📍 New point selected on map: {truth_lat:.6f}, {truth_lon:.6f}")

# Column 2: Google Maps Link & Saving
with col2:
    st.subheader("2. Use Street View for Context")
    
    # Generate Google Maps URL
    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={orig_lat},{orig_lon}"
    # Generate Google Maps street view URL
    street_view_url = f"http://maps.google.com/maps?q=&layer=c&cbll={orig_lat},{orig_lon}&cbp=11,0,0,0,0"

    st.markdown(f"""
    <a href="{street_view_url}" target="_blank">
        <button style="background-color:#4285F4; color:white; padding:10px 24px; border:none; border-radius:4px; cursor:pointer; font-size:16px;">
            👀 Open in Google Street View
        </button>
    </a>
    <br><br>
    <a href="{gmaps_url}" target="_blank">
        <button style="background-color:#34A853; color:white; padding:10px 24px; border:none; border-radius:4px; cursor:pointer; font-size:16px;">
            🗺️ Open standard Google Maps
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.form("save_form"):
        st.write("### 3. Save Label")
        st.write("Click on the map OR enter coordinates manually:")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            manual_lat = st.number_input("Latitude", value=truth_lat if truth_lat is not None else orig_lat, format="%.6f")
        with col_lon:
            manual_lon = st.number_input("Longitude", value=truth_lon if truth_lon is not None else orig_lon, format="%.6f")
        notes = st.text_input("Optional Notes (e.g. 'Inside mall', 'entrance hard to find'):")
        
        submitted = st.form_submit_button("✅ Save Ground Truth & Next")
        
        if submitted:
            # Prefer manual coordinates as they are explicitly in the form
            final_lat = manual_lat
            final_lon = manual_lon
            
            if final_lat == orig_lat and final_lon == orig_lon and notes == "":
                 st.warning("⚠️ You are saving the exact original Overture coordinate. If this is intentional, add a note and click save again, or change the coordinate.")
                 st.stop()

            # Save to CSV
            new_row = pd.DataFrame({
                "id": [props['id']],
                "primary_name": [props.get('primary_name', '')],
                "primary_category": [props.get('primary_category', '')],
                "alternate_categories": [props.get('alternate_categories', '')],
                "source_datasets": [props.get('source_datasets', '')],
                "original_lat": [orig_lat],
                "original_lon": [orig_lon],
                "truth_lat": [final_lat],
                "truth_lon": [final_lon],
                "notes": [notes]
            })

                
            labels_df = pd.concat([labels_df, new_row], ignore_index=True)
            labels_df.to_csv(OUTPUT_CSV, index=False)
                
            # Move to next
            if st.session_state.current_index < total_points - 1:
                st.session_state.current_index += 1
            st.rerun()

    st.write(f"**Current Overture Location:** {orig_lat:.5f}, {orig_lon:.5f}")
