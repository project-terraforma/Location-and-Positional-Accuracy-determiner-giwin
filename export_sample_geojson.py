import pandas as pd
import geopandas as gpd
from shapely import wkt
import json

# Load the parquet file
print("Loading parquet file...")
parquet_path = "/Users/giwin/Documents/CRWN 102(New Attempt)/project_d_samples.parquet"
df = pd.read_parquet(parquet_path)

# Sample 1000 records using a random seed for reproducibility
print(f"Sampling 1000 records from {len(df)} total rows...")
sample_df = df.sample(n=1000, random_state=42).copy()

# Overture's geometry column is usually WKB (Well-Known Binary) or WKT
# Let's check its type to safely convert it to Geopandas geometry
if isinstance(sample_df['geometry'].iloc[0], bytes):
    from shapely import wkb
    sample_df['geometry'] = sample_df['geometry'].apply(lambda x: wkb.loads(x) if x is not None else None)
elif isinstance(sample_df['geometry'].iloc[0], str):
    sample_df['geometry'] = sample_df['geometry'].apply(lambda x: wkt.loads(x) if x is not None else None)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(sample_df, geometry='geometry', crs="EPSG:4326")

# QGIS expects simple properties in GeoJSON (strings, ints, floats)
# We need to flatten the complex columns like 'names', 'categories'
def extract_primary_name(name_col):
    if not name_col:
        return ""
    if isinstance(name_col, dict):
        return name_col.get('primary', "")
    return str(name_col)

def extract_primary_category(cat_col):
    if not cat_col:
        return ""
    if isinstance(cat_col, dict):
        return cat_col.get('primary', "")
    return str(cat_col)

def extract_alternate_categories(cat_col):
    if not cat_col:
        return ""
    if isinstance(cat_col, dict):
        alt = cat_col.get('alternate', [])
        # Join into a single string separated by pipes so CSVs behave nicely
        if isinstance(alt, list):
            return "|".join([str(a) for a in alt if a])
        return str(alt)
    return ""

def extract_sources(source_col):
    if source_col is None:
        return ""
    # sometimes it's a numpy array, we need to convert to list or check len
    try:
        if len(source_col) == 0:
            return ""
    except Exception:
        pass
        
    if isinstance(source_col, (list, tuple)) or hasattr(source_col, '__iter__'):
        # source_col is usually a list of dicts like [{'dataset': 'meta'}, {'dataset': 'microsoft'}]
        datasets = [s.get('dataset', '') for s in source_col if isinstance(s, dict)]
        return "|".join(set([d for d in datasets if d]))
    return str(source_col)

print("Flattening complex columns for QGIS compatibility...")
gdf['primary_name'] = gdf['names'].apply(extract_primary_name)
gdf['primary_category'] = gdf['categories'].apply(extract_primary_category)
gdf['alternate_categories'] = gdf['categories'].apply(extract_alternate_categories)
gdf['source_datasets'] = gdf['sources'].apply(extract_sources)

# Select simple columns to keep file size reasonable and QGIS happy
cols_to_keep = ['id', 'primary_name', 'primary_category', 'alternate_categories', 'source_datasets', 'confidence', 'geometry']
# Also include type or brand if they are simple
if 'type' in gdf.columns:
    cols_to_keep.append('type')
if 'brand' in gdf.columns:
    gdf['brand_name'] = gdf['brand'].apply(lambda x: x.get('names', {}).get('primary', "") if isinstance(x, dict) else str(x))
    cols_to_keep.append('brand_name')

gdf_out = gdf[[c for c in cols_to_keep if c in gdf.columns]]

# Save to GeoJSON
output_path = "/Users/giwin/Documents/CRWN 102(New Attempt)/ground_truth_sample_1000.geojson"
print(f"Saving to {output_path}...")
gdf_out.to_file(output_path, driver="GeoJSON")
print("Done!")
