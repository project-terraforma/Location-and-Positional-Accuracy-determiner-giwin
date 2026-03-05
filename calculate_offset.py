import pandas as pd
import numpy as np

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r

def main():
    print("Loading Ground Truth Dataset...")
    csv_path = "/Users/giwin/Documents/CRWN 102(New Attempt)/ground_truth_labels.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure you run this in the correct directory.")
        return

    print(f"Total labeled points: {len(df)}")
    
    # Drop rows where truth_lat/lon is null just in case
    df = df.dropna(subset=['truth_lat', 'truth_lon'])
    
    print("Calculating Haversine distances for spatial offsets...")
    df['offset_meters'] = haversine_vectorized(
        df['original_lat'], df['original_lon'],
        df['truth_lat'], df['truth_lon']
    )
    
    mean_err = df['offset_meters'].mean()
    median_err = df['offset_meters'].median()
    rmse = np.sqrt((df['offset_meters'] ** 2).mean())
    p90 = np.percentile(df['offset_meters'], 90)
    p95 = np.percentile(df['offset_meters'], 95)
    
    print("\n--- GLOBAL OFFSET METRICS ---")
    print(f"Mean Error (ME): \t{mean_err:.2f} meters")
    print(f"Median Error:    \t{median_err:.2f} meters")
    print(f"RMSE:            \t{rmse:.2f} meters")
    print(f"90th Percentile: \t{p90:.2f} meters")
    print(f"95th Percentile: \t{p95:.2f} meters")
    
    # Save the dataframe with the offset column for analysis
    # Change this to the path you want to save the file to, but keep the file name the same
    out_path = "ground_truth_with_errors.csv"

    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset with calculated offsets to: {out_path}")

if __name__ == "__main__":
    main()