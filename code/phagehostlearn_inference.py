"""
PhageHostLearn - Inference Script

This script offers complete functionality to make predictions for new bacteria, phages or both,
using a trained PhageHostLearn prediction model for Klebsiella phage-host interactions.

Overview:
1. Initial set-up
2. Processing phage genomes and bacterial genomes into RBPs and K-locus proteins
3. Computing feature representations based on ESM-2
4. Predicting new interactions and ranking
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import phagehostlearn_processing as phlp
import phagehostlearn_features as phlf


def main():
    print("=" * 80)
    print("PhageHostLearn - Inference Pipeline")
    print("=" * 80)
    
    # ============================================================================
    # 1. Initial set-up
    # ============================================================================
    print("\n[STEP 1] Initial set-up")
    print("-" * 80)
    
    path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/data'
    code_path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/code'
    phages_path = path + '/phages_genomes'
    bacteria_path = path + '/bacteria_genomes'
    pfam_path = code_path + '/RBPdetect_phageRBPs.hmm'
    xgb_path = code_path + '/RBPdetect_xgb_hmm.json'
    xgb_model_path = code_path + '/phagehostlearn_esm2_xgb.json'
    kaptive_db_path = path + '/Klebsiella_k_locus_primary_reference.gbk'
    suffix = 'inference'
    hmmer_path = path + '/hmmer-3.4'
    
    print(f"Data path: {path}")
    print(f"Phages path: {phages_path}")
    print(f"Bacteria path: {bacteria_path}")
    print(f"Data suffix: {suffix}")
    print("Initial set-up completed")
    
    # ============================================================================
    # 2. Data processing
    # ============================================================================
    print("\n[STEP 2] Data processing")
    print("-" * 80)
    print("The data processing consists of four consecutive steps:")
    print("  (1) Phage gene calling with PHANOTATE")
    print("  (2) Phage protein embedding with bio_embeddings")
    print("  (3) Phage RBP detection")
    print("  (4) Bacterial genome processing with Kaptive")
    
    # 2.1 PHANOTATE
    print("\n[STEP 2.1] Running PHANOTATE on phage genomes...")
    phanotate_path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/.venv/bin/phanotate.py'
    phlp.phanotate_processing(path, phages_path, phanotate_path, data_suffix=suffix, num_phages=2)
    print("PHANOTATE processing completed")

    # Check if RBPbase already exists (if so, we can skip steps 2.2 and 2.3)
    rbpbase_path = os.path.join(path, f'RBPbase{suffix}.csv')
    rbpbase_path_fallback = os.path.join(path, 'RBPbase.csv')
    
    if os.path.exists(rbpbase_path) or os.path.exists(rbpbase_path_fallback):
        print("\n[STEP 2.2] Computing protein embeddings with ProtTransBertBFD...")
        print("  RBPbase already exists - skipping protein embeddings computation.")
        print("\n[STEP 2.3] Running PhageRBPdetect...")
        print("  RBPbase already exists - skipping PhageRBPdetect.")
    else:
        # 2.2 Protein embeddings
        print("\n[STEP 2.2] Computing protein embeddings with ProtTransBertBFD...")
        print("Note: This can be done faster in the cloud (see compute_embeddings_cloud.ipynb)")
        
        embeddings_path = os.path.join(path, f'phage_protein_embeddings{suffix}.csv')
        embeddings_path_fallback = os.path.join(path, 'phage_protein_embeddings.csv')
        
        if os.path.exists(embeddings_path):
            print(f"  Embedding file already exists at: {embeddings_path}")
            print("  Skipping protein embeddings computation.")
            gene_embeddings_file = embeddings_path
        elif os.path.exists(embeddings_path_fallback):
            print(f"  Using existing embedding file (without suffix): {embeddings_path_fallback}")
            print("  Skipping protein embeddings computation.")
            gene_embeddings_file = embeddings_path_fallback
        else:
            print(f"  Embedding file not found. Computing embeddings...")
            phlp.compute_protein_embeddings(path, data_suffix=suffix)
            print("  Protein embeddings computation completed")
            gene_embeddings_file = embeddings_path
        
        # 2.3 PhageRBPdetect
        print("\n[STEP 2.3] Running PhageRBPdetect...")
        phlp.phageRBPdetect(path, pfam_path, hmmer_path, xgb_path, gene_embeddings_file, data_suffix=suffix)
        print("PhageRBPdetect completed")
    
    # 2.4 Kaptive
    print("\n[STEP 2.4] Running Kaptive on bacterial genomes...")
    locibase_path = os.path.join(path, f'Locibase{suffix}.json')
    locibase_path_fallback = os.path.join(path, 'Locibase.json')
    
    if os.path.exists(locibase_path) or os.path.exists(locibase_path_fallback):
        print("  Locibase already exists - skipping Kaptive processing.")
    else:
        phlp.process_bacterial_genomes(path, bacteria_path, kaptive_db_path, data_suffix=suffix)
        print("Kaptive processing completed")
    print("Data processing completed")
    
    # ============================================================================
    # 3. Feature construction
    # ============================================================================
    print("\n[STEP 3] Feature construction")
    print("-" * 80)
    print("Computing ESM-2 embeddings for RBPs and loci proteins")
    print("Note: If ESM-2 embeddings take too long, consider using cloud computing")
    
    # 3.1 ESM-2 features for RBPs
    print("\n[STEP 3.1] Computing ESM-2 embeddings for RBPs...")
    rbp_embeddings_path = path + '/esm2_embeddings_rbp' + suffix + '.csv'
    rbp_embeddings_path_fallback = path + '/esm2_embeddings_rbp.csv'  # Without suffix
    
    if os.path.exists(rbp_embeddings_path):
        print(f"  RBP embeddings file already exists at: {rbp_embeddings_path}")
        print("  Skipping ESM-2 embeddings computation for RBPs.")
    elif os.path.exists(rbp_embeddings_path_fallback):
        print(f"  Using existing RBP embeddings file (without suffix): {rbp_embeddings_path_fallback}")
        rbp_embeddings_path = rbp_embeddings_path_fallback
        print("  Skipping ESM-2 embeddings computation for RBPs.")
    else:
        phlf.compute_esm2_embeddings_rbp(path, data_suffix=suffix)
        print("  ESM-2 embeddings for RBPs completed")
    
    # 3.2 ESM-2 features for loci
    print("\n[STEP 3.2] Computing ESM-2 embeddings for loci...")
    loci_embeddings_path = path + '/esm2_embeddings_loci' + suffix + '.csv'
    loci_embeddings_path_fallback = path + '/esm2_embeddings_loci.csv'  # Without suffix
    
    if os.path.exists(loci_embeddings_path):
        print(f"  Loci embeddings file already exists at: {loci_embeddings_path}")
        print("  Skipping ESM-2 embeddings computation for loci.")
    elif os.path.exists(loci_embeddings_path_fallback):
        print(f"  Using existing loci embeddings file (without suffix): {loci_embeddings_path_fallback}")
        loci_embeddings_path = loci_embeddings_path_fallback
        print("  Skipping ESM-2 embeddings computation for loci.")
    else:
        phlf.compute_esm2_embeddings_loci(path, data_suffix=suffix)
        print("  ESM-2 embeddings for loci completed")
    
    # 3.3 Construct feature matrices
    print("\n[STEP 3.3] Constructing feature matrices...")
    # Use the paths already defined above
    features_esm2, groups_bact = phlf.construct_feature_matrices(
        path, suffix, loci_embeddings_path, rbp_embeddings_path, mode='test'
    )
    print(f"Feature matrices constructed: shape {features_esm2.shape}")
    print("Feature construction completed")
    
    # ============================================================================
    # 4. Predict and rank new interactions
    # ============================================================================
    print("\n[STEP 4] Predicting and ranking new interactions")
    print("-" * 80)
    print("Making predictions per bacterium for all phages")
    print("Using prediction scores to rank potential phages per bacterium")
    
    # 4.1 Load model and make predictions
    print("\n[STEP 4.1] Loading XGBoost model and making predictions...")
    print(f"  Loading model from: {xgb_model_path}")
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"XGBoost model file not found at: {xgb_model_path}")
    xgb = XGBClassifier()
    xgb.load_model(xgb_model_path)
    print("  Model loaded successfully")
    
    scores_xgb = xgb.predict_proba(features_esm2)[:, 1]
    print(f"Predictions made for {len(scores_xgb)} interactions")
    
    # 4.2 Save prediction scores in interaction matrix
    print("\n[STEP 4.2] Saving prediction scores in interaction matrix...")
    groups_bact = np.asarray(groups_bact)
    loci_embeddings = pd.read_csv(loci_embeddings_path)
    rbp_embeddings = pd.read_csv(rbp_embeddings_path)
    bacteria = list(loci_embeddings['accession'])
    phages = list(set(rbp_embeddings['phage_ID']))
    
    print(f"Number of bacteria: {len(bacteria)}")
    print(f"Number of phages: {len(phages)}")
    
    score_matrix = np.zeros((len(bacteria), len(phages)))
    for i, group in enumerate(list(set(groups_bact))):
        scores_this_group = scores_xgb[groups_bact == group]
        score_matrix[i, :] = scores_this_group
    
    results = pd.DataFrame(score_matrix, index=bacteria, columns=phages)
    results_path_csv = path + '/prediction_results' + suffix + '.csv'
    results.to_csv(results_path_csv, index=False)
    print(f"Prediction results saved to: {results_path_csv}")
    
    # 4.3 Rank phages per bacterium
    print("\n[STEP 4.3] Ranking phages per bacterium...")
    ranked = {}
    for group in list(set(groups_bact)):
        scores_this_group = scores_xgb[groups_bact == group]
        ranked_phages = [(x, y) for y, x in sorted(zip(scores_this_group, phages), reverse=True)]
        ranked[bacteria[group]] = ranked_phages
    
    ranked_results_path = path + '/ranked_results' + suffix + '.pickle'
    with open(ranked_results_path, 'wb') as f:
        pickle.dump(ranked, f)
    print(f"Ranked results saved to: {ranked_results_path}")
    print("Prediction and ranking completed")
    
    # ============================================================================
    # 5. Read & interpret results
    # ============================================================================
    print("\n[STEP 5] Reading and interpreting results")
    print("-" * 80)
    
    print("\n[STEP 5.1] Loading ranked results...")
    with open(ranked_results_path, 'rb') as f:
        ranked_results = pickle.load(f)
    print(f"Loaded ranked results for {len(ranked_results)} bacteria")
    
    # 5.2 Print top phages per bacterium
    print("\n[STEP 5.2] Displaying top phages per bacterium...")
    top = 5
    scores = np.zeros((len(ranked_results.keys()), top))
    for i, acc in enumerate(ranked_results.keys()):
        topscores = [round(y, 3) for (x, y) in ranked_results[acc]][:top]
        scores[i, :] = topscores
    
    results_df = pd.DataFrame(scores, index=list(ranked_results.keys()))
    print("\nTop 5 phages per bacterium:")
    print(results_df)
    
    print("\n" + "=" * 80)
    print("Inference pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

