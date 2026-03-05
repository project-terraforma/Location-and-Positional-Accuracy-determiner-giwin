# TruePin
**Prototyping a Spatial Repositioning System for Overture Maps**

## Project Overview
The goal of **TruePin** is to assess and correct the spatial positional accuracy of Points of Interest (POIs) in Overture Maps. While Overture provides frequent global map updates, the geographic "pin" locations for many businesses are misaligned based on real-world routing needs (e.g., placing a pin in the center of an inaccessible mall roof rather than the main parking lot entrance).

This project focuses on three core pillars:
1. **Defining Ground Truth:** Establishing strict edge-case rules for where a pin *should* be placed to optimize user routing.
2. **Measurement:** Building a high-confidence, hand-labeled subset of data to calculate the current spatial offset (Mean, Median, RMSE) of Overture's baseline dataset.
3. **Modeling:** Exploring spatial grid classification (H3) and structural categorization to prototype automated repositioning logic.

---

## The Data

### `project_d_samples.parquet`
The provided raw unannotated sample containing ~5,000 Overture place records used as the base dataset.

### `ground_truth_labels.csv`
A foundational dataset built during this project containing 500 high-confidence, hand-labeled coordinates. This data was built by referencing Google Maps Street View to find precise pedestrian and parking lot entrances.
* **Fields include:** `id`, `primary_name`, `primary_category`, `alternate_categories`, `source_datasets`, original Overture coordinates, and the manually verified Ground Truth coordinates.

### `ground_truth_with_errors.csv`
The analyzed dataset that calculates the exact vector distance between the original Overture pin and the new TruePin coordinate.
* Uses the **Haversine Formula** to compute offset distance in meters (`offset_meters` column).

### `ground_truth_fully_analyzed.csv`
An advanced dataset that uses an LLM (Google Gemini) to read the name and category of all 500 locations and bucket them into four structural types: *Standalone*, *Mall or Nested*, *Large Area*, and *Skyscraper*. This allows for granular error analysis by property architecture.

---

## 🛠️ The Tech Stack & Tools

To rapidly build the initial ground-truth dataset, a custom Python web application was developed.

* **Streamlit & Folium (`label_app.py`):** An interactive UI that loads Overture sample points onto a satellite map. Users can click the map to drop a new pin, pull up Google Street View instantly, and save the verified coordinates directly to CSV.
* **GeoPandas & Pandas:** Used for massive data extraction, coordinate projection, and calculating spatial baseline errors.
* **Google Gemini API (`categorize_and_analyze.py`):** Utilized for structural classification, bridging the gap between raw textual categories and physical real-world building architectures.

## Running the Project

To run any parts of the TruePin pipeline, ensure you have a Python 3 environment running:

```bash
# Set up environment
python3 -m venv OverTureVenv
source OverTureVenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**To start the manual labeling app:**
```bash
streamlit run label_app.py
```

**To calculate spatial offset baseline error:**
```bash
python calculate_offset.py
```

**To categorize locations and calculate error groups via LLM:**
```bash
export GEMINI_API_KEY="your_api_key_here"
python categorize_and_analyze.py
```
