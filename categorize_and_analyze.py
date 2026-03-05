import os
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json
import time
import numpy as np

def main():
    print("Loading offsets dataset...")
    csv_path = "ground_truth_with_errors.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run calculate_offset.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Check if API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please run: export GEMINI_API_KEY='your_api_key_here' in your terminal before running this script.")
        return
        
    client = genai.Client(api_key=api_key)
    
    system_prompt = """
    You are an expert GIS mapping categorizer. Given a Point of Interest (POI) name and its categories, 
    classify its building/structural type into exactly ONE of the following four buckets based on how users would navigate to it:
    
    1. "Standalone" - Individual stores, restaurants, houses, standard offices. (Most common)
    2. "Mall or Nested" - Stores explicitly inside larger structures (e.g., "Starbucks (Target)", mall kiosks, inner shops).
    3. "Large Area" - Massive outdoor locations without a single front door (e.g., Parks, Campgrounds, RV Parks, Golf Courses, Airports).
    4. "Skyscraper" - High-rise buildings with many floors but one primary street lobby.
    
    Output ONLY Valid JSON matching the schema. Do not explain your reasoning.
    """
    
    print("Sending batch to Gemini for categorization... this will take a moment.")
    
    # We will process in batches to avoid rate limits
    batch_size = 50
    structural_types = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"Processing points {i} to {min(i+batch_size, len(df))}...")
        
        # Build prompt for this batch
        prompt = "Classify the following locations:\n\n"
        for idx, row in batch.iterrows():
            prompt += f"ID '{row['id']}': Name: '{row['primary_name']}', Category: '{row['primary_category']}', Alt: '{row['alternate_categories']}'\n"
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    # Return a list of mappings: [ { "id": "...", "category": "..." }, ... ]
                    response_schema={
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING"},
                                "category": {"type": "STRING", "enum": ["Standalone", "Mall or Nested", "Large Area", "Skyscraper"]}
                            },
                            "required": ["id", "category"]
                        }
                    }
                ),
            )
            
            # Parse the JSON response
            results = json.loads(response.text)
            
            # Map back to our list using a dictionary for safety
            result_map = {item['id']: item['category'] for item in results}
            
            for idx, row in batch.iterrows():
                structural_types.append(result_map.get(row['id'], "Standalone")) # fallback to Standalone
                
            time.sleep(2) # rate limit protection
            
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            # Fallback for failed batch
            for _ in range(len(batch)):
                structural_types.append("Error_Fallback")
                
    # Add to dataframe
    df['structural_category'] = structural_types
    
    print("\n--- ERROR METRICS BY STRUCTURAL CATEGORY ---")
    for cat, group in df.groupby('structural_category'):
        print(f"\n{cat.upper()} (n={len(group)}):")
        print(f"  Mean Error: \t{group['offset_meters'].mean():.2f}m")
        print(f"  Median:     \t{group['offset_meters'].median():.2f}m")
        print(f"  RMSE:       \t{np.sqrt((group['offset_meters'] ** 2).mean()):.2f}m")

    print("\n--- ERROR METRICS BY DATA SOURCE ---")
    
    def simplify_source(s):
        s = str(s).lower()
        if 'microsoft' in s and 'meta' in s:
            return 'Both'
        elif 'microsoft' in s:
            return 'Microsoft Only'
        elif 'meta' in s:
            return 'Meta Only'
        return 'Other'
        
    df['source_group'] = df['source_datasets'].apply(simplify_source)
    
    for src, group in df.groupby('source_group'):
        print(f"\n{src.upper()} (n={len(group)}):")
        print(f"  Mean Error: \t{group['offset_meters'].mean():.2f}m")
        print(f"  Median:     \t{group['offset_meters'].median():.2f}m")
        print(f"  RMSE:       \t{np.sqrt((group['offset_meters'] ** 2).mean()):.2f}m")

    out_path = "ground_truth_fully_analyzed.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved final analysis dataset to: {out_path}")

if __name__ == "__main__":
    main()
