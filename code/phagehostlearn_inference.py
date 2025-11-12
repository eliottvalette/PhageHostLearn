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
    phages_path = path + '/phages_genomes'
    bacteria_path = path + '/bacteria_genomes'
    pfam_path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/code/RBPdetect_phageRBPs.hmm'
    xgb_path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/code/RBPdetect_xgb_hmm.json'
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
    print("All phage genomes processed")
    time.sleep(1)
    
    # 2.2 Protein embeddings
    print("\n[STEP 2.2] Computing protein embeddings with ProtTransBertBFD...")
    print("Note: This can be done faster in the cloud (see compute_embeddings_cloud.ipynb)")
    phlp.compute_protein_embeddings(path, data_suffix=suffix)
    print("Protein embeddings computation completed")
    
    # 2.3 PhageRBPdetect
    print("\n[STEP 2.3] Running PhageRBPdetect...")
    gene_embeddings_file = path + '/phage_protein_embeddings' + suffix + '.csv'
    phlp.phageRBPdetect(path, pfam_path, hmmer_path, xgb_path, gene_embeddings_file, data_suffix=suffix)
    print("PhageRBPdetect completed")
    
    # 2.4 Kaptive
    print("\n[STEP 2.4] Running Kaptive on bacterial genomes...")
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
    phlf.compute_esm2_embeddings_rbp(path, data_suffix=suffix)
    print("ESM-2 embeddings for RBPs completed")
    
    # 3.2 ESM-2 features for loci
    print("\n[STEP 3.2] Computing ESM-2 embeddings for loci...")
    phlf.compute_esm2_embeddings_loci(path, data_suffix=suffix)
    print("ESM-2 embeddings for loci completed")
    
    # 3.3 Construct feature matrices
    print("\n[STEP 3.3] Constructing feature matrices...")
    rbp_embeddings_path = path + '/esm2_embeddings_rbp' + suffix + '.csv'
    loci_embeddings_path = path + '/esm2_embeddings_loci' + suffix + '.csv'
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
    xgb = XGBClassifier()
    xgb.load_model('phagehostlearn_esm2_xgb.json')
    print("Model loaded successfully")
    
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

