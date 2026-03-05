import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def evaluate_baseline_heuristic(df):
    """
    Rule-Based Baseline:
    For each location (19 hexes), pick the hex that intersects a building.
    If multiple intersect a building, pick the one closest to a road.
    If none intersect a building, just pick the original center point.
    """
    correct_predictions = 0
    total_locations = df['original_point_id'].nunique()
    
    for loc_id, group in df.groupby('original_point_id'):
        # Get the actual correct hex
        true_hexes = group[group['is_true_pin'] == 1]
        
        # Sometimes 0 hexes caught the true pin (massive outliers), skip
        if true_hexes.empty:
            continue
            
        true_hex = true_hexes.iloc[0]['h3_index']
        
        # --- Baseline Logic ---
        bldg_hexes = group[group['intersects_building'] == 1]
        
        if len(bldg_hexes) > 0:
            # Pick the one closest to a road
            pred_hex = bldg_hexes.sort_values('dist_to_road').iloc[0]['h3_index']
        else:
            # Fallback to the original overture coordinate
            pred_hex = group[group['is_center'] == True].iloc[0]['h3_index']
            
        if pred_hex == true_hex:
            correct_predictions += 1
            
    # Accuracy is (Number of times we guessed the exact right 20m hex) / (Playable points)
    playable = df[df['is_true_pin'] == 1]['original_point_id'].nunique()
    acc = correct_predictions / playable
    print(f"Rule-Based Heuristic Accuracy: {acc:.2%} (Picked the exact correct 20m hexagon)")
    return acc

def train_xgboost(df):
    """
    Train an XGBoost classifier to predict the probability that a given
    hex contains the TruePin, using spatial features.
    """
    print("\nTraining XGBoost Classifier Model...")
    
    # We only care about locations where the true pin actually fell inside our 19-hex grid
    captured_locs = df[df['is_true_pin'] == 1]['original_point_id'].unique()
    play_df = df[df['original_point_id'].isin(captured_locs)].copy()
    
    features = [
        'is_center', 
        'dist_from_center', 
        'intersects_building', 
        'max_building_area', 
        'dist_to_road'
    ]
    
    # One-hot encode the structural category (Standalone, Mall, Skyscraper etc)
    dummies = pd.get_dummies(play_df['struct_cat'], prefix='cat')
    play_df = pd.concat([play_df, dummies], axis=1)
    
    feature_cols = features + list(dummies.columns)
    
    X = play_df[feature_cols]
    y = play_df['is_true_pin']
    
    # Needs to be boolean mapped to int for xgb
    X.loc[:, 'is_center'] = X['is_center'].astype(int)
    
    # 5-Fold Cross Validation so we don't overfit our small dataset
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    # Group by point ID because we are picking 1 hex out of 19 for a specific place
    unique_ids = play_df['original_point_id'].unique()
    
    for train_idx, test_idx in kf.split(unique_ids):
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]
        
        train_mask = play_df['original_point_id'].isin(train_ids)
        test_mask = play_df['original_point_id'].isin(test_ids)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        test_df = play_df[test_mask].copy()
        
        model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4, 
            random_state=42, 
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Predict probabilities of being '1'
        probs = model.predict_proba(X_test)[:, 1]
        test_df['pred_prob'] = probs
        
        correct = 0
        total = len(test_ids)
        
        # For each location, which of its 19 hexes had the highest probability?
        for loc_id, group in test_df.groupby('original_point_id'):
            true_hex = group[group['is_true_pin'] == 1]['h3_index'].values[0]
            pred_hex = group.sort_values('pred_prob', ascending=False).iloc[0]['h3_index']
            
            if pred_hex == true_hex:
                correct += 1
                
        accuracies.append(correct / total)
        
    final_acc = np.mean(accuracies)
    print(f"XGBoost Model Accuracy (5-Fold CV): {final_acc:.2%} (Picked the exact correct 20m hexagon)")
    
    # Feature Importance on full dataset to show what matters
    model.fit(X, y)
    imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Most Important Features for deciding where a pin goes:")
    print(imp.head(5).to_string(index=False))

def main():
    print("Loading H3 classification features dataset...")
    df = pd.read_csv("h3_classification_features.csv")
    
    print("\n--- BASELINE: ALWAYS PICK OVERTURE'S ORIGINAL ---")
    orig_acc = df[df['is_center'] == True]['is_true_pin'].mean()
    print(f"Original Pin Accuracy: {orig_acc:.2%} (The original coordinate was already in the right hex)")
    
    print("\n--- PROTOTYPE 1: RULE-BASED HEURISTIC ---")
    evaluate_baseline_heuristic(df)
    
    print("\n--- PROTOTYPE 2: SUPERVISED MACHINE LEARNING ---")
    train_xgboost(df)

if __name__ == "__main__":
    main()
