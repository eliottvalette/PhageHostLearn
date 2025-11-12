"""
PhageHostLearn - Training Script

This script provides complete functionality to train PhageHostLearn prediction models
for Klebsiella phage-host interactions.

Overview:
1. Initial set-up
2. Data processing (PHANOTATE, protein embeddings, RBP detection, Kaptive, interaction matrix)
3. Feature construction (ESM-2 embeddings)
4. Training and evaluating models (XGBoost with LOGOCV)
5. Results interpretation
"""

import os
import json
import random
import pickle
import subprocess
import time
import math
from os import listdir
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SearchIO import HmmerIO
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve

import torch
import esm
import phagehostlearn_utils as phlu


# ============================================================================
# Data Processing Functions
# ============================================================================

def phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix='', add=False, test=False, num_phages=None):
    """Run PHANOTATE on each phage genome and build the gene database."""
    print(f"  Processing {len(listdir(phage_genomes_path))} phage files...")
    phage_files = listdir(phage_genomes_path)
    print(f'  Number of phage files: {len(phage_files)}')
    if '.DS_Store' in phage_files:
        phage_files.remove('.DS_Store')
    if add:
        rbp_base = pd.read_csv(general_path + '/RBPbase' + data_suffix + '.csv')
        phage_ids = list(set(rbp_base['phage_ID']))
        phage_files = [x for x in phage_files if x.split('.fasta')[0] not in phage_ids]
        print(f'  Processing {len(phage_files)} more phages (add=True)')
    if num_phages is not None:
        print(f'  Processing only the first {num_phages} phages')
        phage_files = phage_files[:num_phages]
    bar = tqdm(total=len(phage_files), position=0, leave=True, desc='  Processing phage genomes')
    name_list = []
    gene_list = []
    gene_ids = []

    for file in phage_files:
        count = 1
        file_dir = phage_genomes_path + '/' + file
        raw_str = phanotate_path + ' ' + file_dir
        process = subprocess.Popen(raw_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        stdout_text = stdout.decode('utf-8', errors='ignore')
        if process.returncode != 0:
            raise RuntimeError(
                f"PHANOTATE command failed for {file_dir} with exit code {process.returncode}. "
                f"Command output:\n{stdout_text}"
            )
        std_splits = stdout.split(sep=b'\n')[2:]
        if not any(split.strip() for split in std_splits):
            raise ValueError(
                f"PHANOTATE did not return ORF predictions for {file_dir}. "
                f"Command output:\n{stdout_text}"
            )

        temp_tab_path = os.path.join(general_path, 'phage_results.tsv')
        with open(temp_tab_path, 'wb') as temp_tab:
            for split in std_splits:
                split = split.replace(b',', b'')
                temp_tab.write(split + b'\n')
        try:
            results_orfs = pd.read_csv(temp_tab_path, sep='\t', lineterminator='\n', index_col=False)
        except pd.errors.EmptyDataError as exc:
            with open(temp_tab_path, 'r', encoding='utf-8', errors='ignore') as temp_in:
                temp_preview = temp_in.read()
            raise ValueError(
                f"PHANOTATE output for {file_dir} produced an empty or invalid TSV file.\n"
                f"Command output:\n{stdout_text}\n"
                f"Temporary TSV content:\n{temp_preview}"
            ) from exc

        name = file.split('.fasta')[0]
        sequence = str(SeqIO.read(file_dir, 'fasta').seq)
        for j, strand in enumerate(results_orfs['FRAME']):
            start = results_orfs['#START'][j]
            stop = results_orfs['STOP'][j]
            if strand == '+':
                gene = sequence[start - 1:stop]
            else:
                sequence_part = sequence[stop - 1:start]
                gene = str(Seq(sequence_part).reverse_complement())
            name_list.append(name)
            gene_list.append(gene)
            gene_ids.append(name + '_gp' + str(count))
            count += 1
        bar.update(1)
    bar.close()

    if not test and os.path.exists(os.path.join(general_path, 'phage_results.tsv')):
        os.remove(os.path.join(general_path, 'phage_results.tsv'))

    genebase = pd.DataFrame(list(zip(name_list, gene_ids, gene_list)), columns=['phage_ID', 'gene_ID', 'gene_sequence'])
    if add:
        old_genebase = pd.read_csv(general_path + '/phage_genes' + data_suffix + '.csv')
        genebase = pd.concat([old_genebase, genebase], axis=0)
    genebase.to_csv(general_path + '/phage_genes' + data_suffix + '.csv', index=False)
    print(f'  Completed PHANOTATE - Number of phage genes: {len(genebase)}')
    return


def compute_protein_embeddings(general_path, data_suffix='', add=False, num_genes=None, force_recompute=False):
    """Compute protein embeddings using ProtTransBertBFD."""
    embeddings_path = os.path.join(general_path, f'phage_protein_embeddings{data_suffix}.csv')
    embeddings_path_fallback = os.path.join(general_path, 'phage_protein_embeddings.csv')
    
    print(f'  Checking for existing embeddings file: {embeddings_path}')
    if (not force_recompute) and os.path.exists(embeddings_path):
        print(f'  Embedding file already exists at {embeddings_path}. Skipping computation.')
        print('  Use force_recompute=True to rebuild it.')
        return
    elif (not force_recompute) and os.path.exists(embeddings_path_fallback):
        print(f'  Using existing embedding file (without suffix): {embeddings_path_fallback}')
        print('  Skipping computation.')
        return
    
    print(f'{"Force recompute" if force_recompute else "File not found"}. Computing embeddings...')
    
    genebase = pd.read_csv(general_path + '/phage_genes' + data_suffix + '.csv')
    if num_genes is not None:
        print(f'  Processing only the first {num_genes} phage genes')
        genebase = genebase.head(num_genes)
    print(f'  Number of phage genes: {len(genebase)}')
    
    print('  Importing ProtTransBertBFDEmbedder...')
    from bio_embeddings.embed import ProtTransBertBFDEmbedder
    print('  Import successful')
    
    time_start = time.time()
    embedder = ProtTransBertBFDEmbedder()
    time_end = time.time()
    print(f'  Time taken to initialize embedder: {time_end - time_start:.2f} seconds')
    print('  Embedder initialized')
    if add:
        print('  Adding new protein embeddings')
        old_embeddings_df = pd.read_csv(embeddings_path)
        protein_ids = list(old_embeddings_df['ID'])
        sequences = []
        names = []
        for i, sequence in enumerate(genebase['gene_sequence']):
            if genebase['gene_ID'][i] not in protein_ids:
                sequences.append(str(Seq(sequence).translate())[:-1])
                names.append(genebase['gene_ID'][i])
    else:
        print('  Computing protein embeddings for all phage genes')
        names = list(genebase['gene_ID'])
        print(f'  Number of protein sequences to embed: {len(names)}')
        sequences = [str(Seq(sequence).translate())[:-1] for sequence in genebase['gene_sequence']]

    print(f'  Number of protein sequences to embed: {len(sequences)}')
    protein_embeddings = []
    progress_bar = tqdm(sequences, desc='  Computing protein embeddings', unit='protein')
    for protein_sequence in progress_bar:
        reduced_embedding = embedder.reduce_per_protein(embedder.embed(protein_sequence))
        protein_embeddings.append(reduced_embedding)
    embeddings_df = pd.concat([pd.DataFrame({'ID': names}), pd.DataFrame(protein_embeddings)], axis=1)
    if add:
        embeddings_df = pd.DataFrame(np.vstack([old_embeddings_df, embeddings_df]), columns=old_embeddings_df.columns)
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f'  Protein embeddings saved to: {embeddings_path}')
    return


def hmmpress_python(hmm_path, pfam_file):
    """Press a profiles database, necessary to do scanning."""
    cd_str = 'cd ' + hmm_path
    press_str = 'hmmpress ' + pfam_file
    command = cd_str + '; ' + press_str
    press_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()
    return press_out, press_err


def single_hmmscan_python(hmm_path, pfam_file, fasta_file):
    """Run hmmscan for a given FASTA file of one (or multiple) sequences."""
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_process.communicate()

    scan_str = 'hmmscan ' + pfam_file + ' ' + fasta_file + ' > hmmscan_out.txt'
    scan_process = subprocess.Popen(scan_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    scan_process.communicate()

    with open('hmmscan_out.txt') as results_handle:
        scan_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmscan_out.txt')
    return scan_res


def hmmscan_python(hmm_path, pfam_file, sequences_file, threshold=18):
    """Scan sequences for domains using hmmscan."""
    domains = []
    scores = []
    biases = []
    ranges = []
    for sequence in SeqIO.parse(sequences_file, 'fasta'):
        with open('single_sequence.fasta', 'w') as temp_fasta:
            temp_fasta.write('>' + sequence.id + '\n' + str(sequence.seq) + '\n')

        scan_res = single_hmmscan_python(hmm_path, pfam_file, 'single_sequence.fasta')
        for line in scan_res:
            try:
                for hit in line.hits:
                    hsp = hit._items[0]
                    aln_start = hsp.query_range[0]
                    aln_stop = hsp.query_range[1]
                    if (hit.bitscore >= threshold) and (hit.id not in domains):
                        domains.append(hit.id)
                        scores.append(hit.bitscore)
                        biases.append(hit.bias)
                        ranges.append((aln_start, aln_stop))
            except IndexError:
                pass
    os.remove('single_sequence.fasta')
    return domains, scores, biases, ranges


def gene_domain_scan(hmmpath, pfam_file, gene_hits, threshold=18):
    """Run hmmscan on translated gene hits."""
    with open('protein_hits.fasta', 'w') as hits_fasta:
        for i, gene_hit in enumerate(gene_hits):
            protein_sequence = str(Seq(gene_hit).translate())[:-1]
            hits_fasta.write('>' + str(i) + '_proteindomain_hit\n' + protein_sequence + '\n')
    domains, scores, biases, ranges = hmmscan_python(hmmpath, pfam_file, 'protein_hits.fasta', threshold)
    os.remove('protein_hits.fasta')
    return domains, scores, biases, ranges


def phageRBPdetect(general_path, pfam_path, hmmer_path, xgb_path, gene_embeddings_path, data_suffix=''):
    """Detect receptor-binding proteins using PhageRBPdetect."""
    print(f"  Loading genebase from {general_path + '/phage_genes' + data_suffix + '.csv'}")
    genebase = pd.read_csv(general_path + '/phage_genes' + data_suffix + '.csv')
    new_blocks = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 'Head_binding', 'DUF3751',
                  'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 'DUF2163', 'Phage_fiber_2', 'unknown_N0',
                  'unknown_N1', 'unknown_N2', 'unknown_N3', 'unknown_N4', 'unknown_N6', 'unknown_N10', 'unknown_N11',
                  'unknown_N12', 'unknown_N13', 'unknown_N17', 'unknown_N19', 'unknown_N23', 'unknown_N24',
                  'unknown_N26', 'unknown_N29', 'unknown_N36', 'unknown_N45', 'unknown_N48', 'unknown_N49',
                  'unknown_N53', 'unknown_N57', 'unknown_N60', 'unknown_N61', 'unknown_N65', 'unknown_N73',
                  'unknown_N82', 'unknown_N83', 'unknown_N101', 'unknown_N114', 'unknown_N119', 'unknown_N122',
                  'unknown_N163', 'unknown_N174', 'unknown_N192', 'unknown_N200', 'unknown_N206', 'unknown_N208',
                  'Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel',
                  'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 'gp12-short_mid', 'Collar',
                  'unknown_C2', 'unknown_C3', 'unknown_C8', 'unknown_C15', 'unknown_C35', 'unknown_C54', 'unknown_C76',
                  'unknown_C100', 'unknown_C105', 'unknown_C112', 'unknown_C123', 'unknown_C179', 'unknown_C201',
                  'unknown_C203', 'unknown_C228', 'unknown_C234', 'unknown_C242', 'unknown_C258', 'unknown_C262',
                  'unknown_C267', 'unknown_C268', 'unknown_C274', 'unknown_C286', 'unknown_C292', 'unknown_C294',
                  'Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

    print("  Pressing HMM database...")
    output, err = hmmpress_python(hmmer_path, pfam_path)
    print(output.decode('utf-8', errors='ignore'))

    phage_genes = genebase['gene_sequence']
    hmm_scores = {item: [0] * len(phage_genes) for item in new_blocks}
    bar = tqdm(total=len(phage_genes), position=0, leave=True, desc='  Scanning phage genes')
    for i, sequence in enumerate(phage_genes):
        hits, scores, biases, ranges = gene_domain_scan(hmmer_path, pfam_path, [sequence])
        for j, dom in enumerate(hits):
            hmm_scores[dom][i] = scores[j]
        bar.update(1)
    bar.close()

    print("  Loading embeddings and computing features...")
    embeddings_df = pd.read_csv(gene_embeddings_path)
    embeddings = np.asarray(embeddings_df.iloc[:, 1:])
    hmm_scores_array = np.asarray(pd.DataFrame(hmm_scores))
    features = np.concatenate((embeddings, hmm_scores_array), axis=1)

    print("  Loading XGBoost model for RBP detection...")
    xgb_saved = XGBClassifier()
    xgb_saved.load_model(xgb_path)

    print("  Making predictions...")
    score_xgb = xgb_saved.predict_proba(features)[:, 1]
    preds_xgb = (score_xgb > 0.5) * 1

    rbp_base = {'phage_ID': [], 'protein_ID': [], 'protein_sequence': [], 'dna_sequence': [], 'xgb_score': []}
    for i, dna_sequence in enumerate(genebase['gene_sequence']):
        if preds_xgb[i] == 1:
            rbp_base['phage_ID'].append(genebase['phage_ID'][i])
            rbp_base['protein_ID'].append(genebase['gene_ID'][i])
            rbp_base['protein_sequence'].append(str(Seq(dna_sequence).translate())[:-1])
            rbp_base['dna_sequence'].append(dna_sequence)
            rbp_base['xgb_score'].append(score_xgb[i])
    rbp_base_df = pd.DataFrame(rbp_base)
    to_delete = [i for i, protein_seq in enumerate(rbp_base_df['protein_sequence']) if (len(protein_seq) < 200 or len(protein_seq) > 1500)]
    rbp_base_df = rbp_base_df.drop(to_delete).reset_index(drop=True)
    rbp_base_df.to_csv(general_path + '/RBPbase' + data_suffix + '.csv', index=False)
    print(f"  RBP detection completed - Found {len(rbp_base_df)} RBPs")
    return


def kaptive_python(database_path, file_path, output_path):
    """Wrapper for the Kaptive command-line call."""
    command = 'python kaptive.py -a ' + file_path + ' -k ' + database_path + ' -o ' + output_path + '/ --no_table'
    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssprocess.communicate()
    return


def process_bacterial_genomes(general_path, bact_genomes_path, database_path, data_suffix='', add=False):
    """Process bacterial genomes with Kaptive to extract K-locus proteins."""
    fastas = listdir(bact_genomes_path)
    try:
        fastas.remove('.DS_Store')
    except ValueError:
        pass
    if add:
        with open(general_path + '/Locibase' + data_suffix + '.json') as dict_file:
            old_locibase = json.load(dict_file)
        loci_accessions = list(old_locibase.keys())
        fastas = [x for x in fastas if x.split('.fasta')[0] not in loci_accessions]
        print(f'  Processing {len(fastas)} more bacteria (add=True)')
    accessions = [file.split('.fasta')[0] for file in fastas]
    serotypes = []
    loci_results = {}
    pbar = tqdm(total=len(fastas), desc='  Processing bacterial genomes')
    with open(general_path + '/kaptive_results_all_loci.fasta', 'w') as big_fasta:
        for i, file in enumerate(fastas):
            file_path = bact_genomes_path + '/' + file
            kaptive_python(database_path, file_path, general_path)

            results = json.load(open(general_path + '/kaptive_results.json'))
            serotypes.append(results[0]['Best match']['Type'])
            for gene in results[0]['Locus genes']:
                try:
                    protein = gene['tblastn result']['Protein sequence']
                    protein = protein.replace('-', '').replace('*', '')
                except KeyError:
                    protein = gene['Reference']['Protein sequence']
                loci_results.setdefault(accessions[i], []).append(protein[:-1])

            loci_sequence = ''
            for record in SeqIO.parse(general_path + '/kaptive_results_' + file, 'fasta'):
                loci_sequence += str(record.seq)
            big_fasta.write('>' + accessions[i] + '\n' + loci_sequence + '\n')

            for extension in ['.ndb', '.not', '.ntf', '.nto']:
                os.remove(file_path + extension)
            os.remove(general_path + '/kaptive_results.json')
            os.remove(general_path + '/kaptive_results_' + file)
            pbar.update(1)
    pbar.close()

    sero_df = pd.DataFrame(serotypes, columns=['sero'])
    if add:
        loci_results = {**old_locibase, **loci_results}
        old_seros = pd.read_csv(general_path + '/serotypes' + data_suffix + '.csv')
        sero_df = pd.concat([old_seros, sero_df], axis=0)
    sero_df.to_csv(general_path + '/serotypes' + data_suffix + '.csv', index=False)
    with open(general_path + '/Locibase' + data_suffix + '.json', 'w') as dict_file:
        json.dump(loci_results, dict_file)
    print(f"  Kaptive processing completed - Processed {len(accessions)} bacteria")
    return


def xlsx_database_to_csv(xlsx_file, save_path, index_col=0, header=0, export=True):
    """Convert an XLSX interaction matrix to CSV."""
    interactions_matrix = pd.read_excel(xlsx_file, index_col=index_col, header=header)
    if export:
        interactions_matrix.to_csv(save_path + '.csv')
        return
    return interactions_matrix


def process_interactions(general_path, interactions_xlsx_path, data_suffix=''):
    """Process the interaction matrix and export it to CSV."""
    output_path = os.path.join(general_path, f'phage_host_interactions{data_suffix}.csv')
    output_path_fallback = os.path.join(general_path, 'phage_host_interactions.csv')
    
    if os.path.exists(output_path):
        print(f"  Interaction matrix already exists at {output_path}. Skipping processing.")
        return
    elif os.path.exists(output_path_fallback):
        print(f"  Using existing interaction matrix (without suffix): {output_path_fallback}")
        print("  Skipping processing.")
        return
    
    if not os.path.exists(interactions_xlsx_path):
        raise FileNotFoundError(
            f"Interaction matrix XLSX file not found at {interactions_xlsx_path} "
            f"and CSV file not found at {output_path} or {output_path_fallback}. "
            f"Please provide the XLSX file or ensure the CSV file exists."
        )
    
    output = os.path.join(general_path, f'phage_host_interactions{data_suffix}')
    xlsx_database_to_csv(interactions_xlsx_path, output)
    print(f"  Interaction matrix processed and saved to: {output}.csv")
    return


# ============================================================================
# Feature Construction Functions
# ============================================================================

def compute_esm2_embeddings_rbp(general_path, data_suffix='', add=False, force_recompute=False):
    """
    This function computes ESM-2 embeddings for the RBPs, from the RBPbase.csv file.
    """
    embeddings_path = os.path.join(general_path, f'esm2_embeddings_rbp{data_suffix}.csv')
    embeddings_path_fallback = os.path.join(general_path, 'esm2_embeddings_rbp.csv')
    
    print(f'  Checking for existing embeddings file: {embeddings_path}')
    if (not force_recompute) and os.path.exists(embeddings_path):
        print(f'  RBP embeddings file already exists at {embeddings_path}. Skipping computation.')
        print('  Use force_recompute=True to rebuild it.')
        return
    elif (not force_recompute) and os.path.exists(embeddings_path_fallback):
        print(f'  Using existing RBP embeddings file (without suffix): {embeddings_path_fallback}')
        print('  Skipping computation.')
        return
    
    print("  Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    print("  Loading RBPbase...")
    RBPbase = pd.read_csv(general_path+'/RBPbase'+data_suffix+'.csv')
    if add == True:
        old_embeddings_df = pd.read_csv(general_path+'/esm2_embeddings_rbp'+data_suffix+'.csv')
        protein_ids = list(set(old_embeddings_df['protein_ID']))
        to_delete = [i for i, prot_id in enumerate(RBPbase['protein_ID']) if prot_id in protein_ids]
        RBPbase = RBPbase.drop(to_delete)
        RBPbase = RBPbase.reset_index(drop=True)
        print(f'  Processing {len(RBPbase["protein_sequence"])} more sequences (add=True)')

    print(f"  Computing embeddings for {len(RBPbase['protein_sequence'])} RBPs...")
    bar = tqdm(total=len(RBPbase['protein_sequence']), position=0, leave=True)
    sequence_representations = []
    for i, sequence in enumerate(RBPbase['protein_sequence']):
        data = [(RBPbase['protein_ID'][i], sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        for j, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[j, 1 : len(seq) + 1].mean(0))
        bar.update(1)
    bar.close()

    print("  Saving results...")
    phage_ids = RBPbase['phage_ID']
    ids = RBPbase['protein_ID']
    embeddings_df = pd.concat([pd.DataFrame({'phage_ID':phage_ids}), pd.DataFrame({'protein_ID':ids}), pd.DataFrame(sequence_representations).astype('float')], axis=1)
    if add == True:
        embeddings_df = pd.DataFrame(np.vstack([old_embeddings_df, embeddings_df]), columns=old_embeddings_df.columns)
    embeddings_path = general_path+'/esm2_embeddings_rbp'+data_suffix+'.csv'
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f"  ESM-2 embeddings for RBPs saved to: {embeddings_path}")
    return


def compute_esm2_embeddings_loci(general_path, data_suffix='', add=False, force_recompute=False):
    """
    This function computes ESM-2 embeddings for the loci proteins, from the Locibase.json file.
    """
    embeddings_path = os.path.join(general_path, f'esm2_embeddings_loci{data_suffix}.csv')
    embeddings_path_fallback = os.path.join(general_path, 'esm2_embeddings_loci.csv')
    
    print(f'  Checking for existing embeddings file: {embeddings_path}')
    if (not force_recompute) and os.path.exists(embeddings_path):
        print(f'  Loci embeddings file already exists at {embeddings_path}. Skipping computation.')
        print('  Use force_recompute=True to rebuild it.')
        return
    elif (not force_recompute) and os.path.exists(embeddings_path_fallback):
        print(f'  Using existing loci embeddings file (without suffix): {embeddings_path_fallback}')
        print('  Skipping computation.')
        return
    
    print("  Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    print("  Loading Locibase...")
    dict_file = open(general_path+'/Locibase'+data_suffix+'.json')
    loci_dict = json.load(dict_file)
    if add == True:
        old_embeddings_df = pd.read_csv(general_path+'/esm2_embeddings_loci'+data_suffix+'.csv')
        old_accessions = list(set(old_embeddings_df['accession']))
        for key in loci_dict.keys():
            if key in old_accessions:
                del loci_dict[key]
        print(f'  Processing {len(loci_dict.keys())} more bacteria (add=True)')

    print(f"  Computing embeddings for {len(loci_dict.keys())} loci...")
    loci_representations = []
    for key in tqdm(loci_dict.keys()):
        embeddings = []
        for sequence in loci_dict[key]:
            data = [(key, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            for i, (_, seq) in enumerate(data):
                embeddings.append(token_representations[i, 1 : len(seq) + 1].mean(0))
        locus_embedding = np.mean(np.vstack(embeddings), axis=0)
        loci_representations.append(locus_embedding)

    print("  Saving results...")
    embeddings_df = pd.concat([pd.DataFrame({'accession':list(loci_dict.keys())}), pd.DataFrame(loci_representations)], axis=1)
    if add == True:
        embeddings_df = pd.DataFrame(np.vstack([old_embeddings_df, embeddings_df]), columns=old_embeddings_df.columns)
    embeddings_path = general_path+'/esm2_embeddings_loci'+data_suffix+'.csv'
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f"  ESM-2 embeddings for loci saved to: {embeddings_path}")
    return


def construct_feature_matrices(path, suffix, lociembeddings_path, rbpembeddings_path, mode='train'):
    """
    This function constructs two corresponding feature matrices ready for machine learning, 
    starting from the ESM-2 embeddings of RBPs and loci proteins.
    """
    print("  Loading embeddings...")
    RBP_embeddings = pd.read_csv(rbpembeddings_path)
    loci_embeddings = pd.read_csv(lociembeddings_path)
    if mode == 'train':
        interactions = pd.read_csv(path+'/phage_host_interactions'+suffix+'.csv', index_col=0)

    print("  Constructing multi-RBP representations...")
    multi_embeddings = []
    names = []
    for phage_id in list(set(RBP_embeddings['phage_ID'])):
        rbp_embeddings = RBP_embeddings.iloc[:,2:][RBP_embeddings['phage_ID'] == phage_id]
        multi_embedding = np.mean(np.asarray(rbp_embeddings), axis=0)
        names.append(phage_id)
        multi_embeddings.append(multi_embedding)
    multiRBP_embeddings = pd.concat([pd.DataFrame({'phage_ID': names}), pd.DataFrame(multi_embeddings)], axis=1)

    print("  Constructing feature matrices...")
    features_lan = []
    labels = []
    groups_loci = []
    groups_phage = []

    for i, accession in enumerate(loci_embeddings['accession']):
        for j, phage_id in enumerate(multiRBP_embeddings['phage_ID']):
            if mode == 'train':
                interaction = interactions.loc[accession][phage_id]
                if math.isnan(interaction) == False:
                    features_lan.append(pd.concat([loci_embeddings.iloc[i, 1:], multiRBP_embeddings.iloc[j, 1:]]))
                    labels.append(int(interaction))
                    groups_loci.append(i)
                    groups_phage.append(j)
            elif mode == 'test':
                features_lan.append(pd.concat([loci_embeddings.iloc[i, 1:], multiRBP_embeddings.iloc[j, 1:]]))
                groups_loci.append(i)
                groups_phage.append(j)

    features_lan = np.asarray(features_lan)
    print(f"  Dimensions match? {features_lan.shape[1] == (loci_embeddings.shape[1]+multiRBP_embeddings.shape[1]-2)}")
    print(f"  Feature matrix shape: {features_lan.shape}")

    if mode == 'train':
        print(f"  Number of labels: {len(labels)}")
        return features_lan, labels, groups_loci, groups_phage
    elif mode == 'test':
        return features_lan, groups_loci


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("PhageHostLearn - Training Pipeline")
    print("=" * 80)
    
    # ============================================================================
    # 1. Initial set-up
    # ============================================================================
    print("\n[STEP 1] Initial set-up")
    print("-" * 80)
    
    script_directory = Path(__file__).resolve().parent
    project_root = str(script_directory.parent)
    general_path = os.path.join(project_root, 'data')
    results_path = os.path.join(project_root, 'results')
    data_suffix = ''
    
    print(f"Project root: {project_root}")
    print(f"General path: {general_path}")
    print(f"Results path: {results_path}")
    print(f"Data suffix: {data_suffix}")
    
    # ============================================================================
    # 2. Data processing
    # ============================================================================
    print("\n[STEP 2] Data processing")
    print("-" * 80)
    print("The data processing consists of five consecutive steps:")
    print("  (1) Phage gene calling with PHANOTATE")
    print("  (2) Phage protein embedding with bio_embeddings")
    print("  (3) Phage RBP detection")
    print("  (4) Bacterial genome processing with Kaptive")
    print("  (5) Processing the interaction matrix")
    
    # 2.1 PHANOTATE
    print("\n[STEP 2.1] Running PHANOTATE on phage genomes...")
    phage_genomes_path = os.path.join(general_path, 'phages_genomes')
    phanotate_path = '/Users/eliottvalette/Documents/Clones/PhageHostLearn/.venv/bin/phanotate.py'
    phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix=data_suffix, num_phages=2)
    
    # Check if RBPbase already exists (if so, we can skip steps 2.2 and 2.3)
    rbpbase_path = os.path.join(general_path, f'RBPbase{data_suffix}.csv')
    rbpbase_path_fallback = os.path.join(general_path, 'RBPbase.csv')
    
    if os.path.exists(rbpbase_path) or os.path.exists(rbpbase_path_fallback):
        print("\n[STEP 2.2] Computing protein embeddings with ProtTransBertBFD...")
        print("  RBPbase already exists - skipping protein embeddings computation.")
        print("\n[STEP 2.3] Running PhageRBPdetect...")
        print("  RBPbase already exists - skipping PhageRBPdetect.")
    else:
        # 2.2 Protein embeddings
        print("\n[STEP 2.2] Computing protein embeddings with ProtTransBertBFD...")
        compute_protein_embeddings(general_path, data_suffix=data_suffix, num_genes=5)
        
        # 2.3 PhageRBPdetect
        print("\n[STEP 2.3] Running PhageRBPdetect...")
        pfam_path = os.path.join(general_path, 'RBPdetect_phageRBPs.hmm')
        if not os.path.exists(pfam_path):
            pfam_path = os.path.join(project_root, 'code', 'RBPdetect_phageRBPs.hmm')
        hmmer_path = '/Users/Dimi/hmmer-3.3.1'
        xgb_path = os.path.join(general_path, 'RBPdetect_xgb_hmm.json')
        if not os.path.exists(xgb_path):
            xgb_path = os.path.join(project_root, 'code', 'RBPdetect_xgb_hmm.json')
        gene_embeddings_path = os.path.join(general_path, f'phage_protein_embeddings{data_suffix}.csv')
        gene_embeddings_path_fallback = os.path.join(general_path, 'phage_protein_embeddings.csv')
        if not os.path.exists(gene_embeddings_path) and os.path.exists(gene_embeddings_path_fallback):
            gene_embeddings_path = gene_embeddings_path_fallback
        phageRBPdetect(general_path, pfam_path, hmmer_path, xgb_path, gene_embeddings_path, data_suffix=data_suffix)
    
    # 2.4 Kaptive
    print("\n[STEP 2.4] Running Kaptive on bacterial genomes...")
    locibase_path = os.path.join(general_path, f'Locibase{data_suffix}.json')
    locibase_path_fallback = os.path.join(general_path, 'Locibase.json')
    
    if os.path.exists(locibase_path) or os.path.exists(locibase_path_fallback):
        print("  Locibase already exists - skipping Kaptive processing.")
    else:
        bact_genomes_path = os.path.join(general_path, 'klebsiella_genomes', 'fasta_files')
        kaptive_database_path = os.path.join(general_path, 'Klebsiella_k_locus_primary_reference.gbk')
        process_bacterial_genomes(general_path, bact_genomes_path, kaptive_database_path, data_suffix=data_suffix)
    
    # 2.5 Process interactions
    print("\n[STEP 2.5] Processing interaction matrix...")
    interactions_xlsx_path = os.path.join(general_path, 'klebsiella_phage_host_interactions.xlsx')
    process_interactions(general_path, interactions_xlsx_path, data_suffix=data_suffix)
    
    print("\nData processing completed")
    
    # ============================================================================
    # 3. Feature construction
    # ============================================================================
    print("\n[STEP 3] Feature construction")
    print("-" * 80)
    print("Computing ESM-2 embeddings for RBPs and loci proteins")
    print("Note: If ESM-2 embeddings take too long, consider using cloud computing")
    
    # 3.1 ESM-2 features for RBPs
    print("\n[STEP 3.1] Computing ESM-2 embeddings for RBPs...")
    compute_esm2_embeddings_rbp(general_path, data_suffix=data_suffix)
    
    # 3.2 ESM-2 features for loci
    print("\n[STEP 3.2] Computing ESM-2 embeddings for loci...")
    compute_esm2_embeddings_loci(general_path, data_suffix=data_suffix)
    
    # 3.3 Construct feature matrices
    print("\n[STEP 3.3] Constructing feature matrices...")
    rbp_embeddings_path = os.path.join(general_path, f'esm2_embeddings_rbp{data_suffix}.csv')
    loci_embeddings_path = os.path.join(general_path, f'esm2_embeddings_loci{data_suffix}.csv')
    
    # Use fallback paths if suffixed files don't exist
    if not os.path.exists(rbp_embeddings_path):
        rbp_embeddings_path = os.path.join(general_path, 'esm2_embeddings_rbp.csv')
    if not os.path.exists(loci_embeddings_path):
        loci_embeddings_path = os.path.join(general_path, 'esm2_embeddings_loci.csv')
    features_esm2, labels, groups_loci, groups_phage = construct_feature_matrices(
        general_path, data_suffix, loci_embeddings_path, rbp_embeddings_path, mode='train'
    )
    
    print("\nFeature construction completed")
    
    # ============================================================================
    # 4. Training and evaluating models
    # ============================================================================
    print("\n[STEP 4] Training and evaluating models")
    print("-" * 80)
    
    # 4.1 Training models
    print("\n[STEP 4.1] Training XGBoost model...")
    cpus = 6
    labels = np.asarray(labels)
    
    imbalance = sum([1 for i in labels if i==1]) / sum([1 for i in labels if i==0])
    print(f"  Class imbalance ratio: {imbalance:.3f}")
    
    xgb = XGBClassifier(
        scale_pos_weight=1/imbalance,
        learning_rate=0.3,
        n_estimators=250,
        max_depth=7,
        n_jobs=cpus,
        eval_metric='logloss'
    )
    print("  Fitting XGBoost model...")
    xgb.fit(features_esm2, labels)
    
    model_path = 'phagehostlearn_vbeta.json'
    xgb.save_model(model_path)
    print(f"  Model saved to: {model_path}")
    
    # 4.2 LOGOCV evaluation
    print("\n[STEP 4.2] Running Leave-One-Group-Out Cross-Validation (LOGOCV)...")
    
    print("  Loading similarity matrix and setting threshold...")
    matrix = np.loadtxt(general_path + '/all_loci_score_matrix.txt', delimiter='\t')
    threshold = 0.995
    threshold_str = '995'
    group_i = 0
    new_groups = [np.nan] * len(groups_loci)
    for i in range(matrix.shape[0]):
        cluster = np.where(matrix[i, :] >= threshold)[0]
        oldgroups_i = [k for k, x in enumerate(groups_loci) if x in cluster]
        if np.isnan(new_groups[groups_loci.index(i)]):
            for ogi in oldgroups_i:
                new_groups[ogi] = group_i
            group_i += 1
    groups_loci = new_groups
    print(f'  Number of unique groups: {len(set(groups_loci))}')
    
    print("  Running LOGOCV...")
    logo = LeaveOneGroupOut()
    scores_lan = []
    label_list = []
    labels = np.asarray(labels)
    pbar = tqdm(total=len(set(groups_loci)), desc='  LOGOCV iterations')
    for train_index, test_index in logo.split(features_esm2, labels, groups_loci):
        Xlan_train, Xlan_test = features_esm2[train_index], features_esm2[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        imbalance = sum([1 for i in y_train if i==1]) / sum([1 for i in y_train if i==0])

        xgb = XGBClassifier(
            scale_pos_weight=1/imbalance,
            learning_rate=0.3,
            n_estimators=250,
            max_depth=7,
            n_jobs=cpus,
            eval_metric='logloss'
        )
        xgb.fit(Xlan_train, y_train)
        score_xgb = xgb.predict_proba(Xlan_test)[:, 1]
        scores_lan.append(score_xgb)
        label_list.append(y_test)
        pbar.update(1)
    pbar.close()
    
    print("  Saving LOGOCV results...")
    logo_results = {'labels': label_list, 'scores_language': scores_lan}
    results_file = results_path + '/v3.4/combined_logocv_results_v34_' + threshold_str + '.pickle'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'wb') as f:
        pickle.dump(logo_results, f)
    print(f"  LOGOCV results saved to: {results_file}")
    
    # ============================================================================
    # 5. Results interpretation
    # ============================================================================
    print("\n[STEP 5] Results interpretation")
    print("-" * 80)
    
    print("  Loading LOGOCV results...")
    with open(results_file, 'rb') as f:
        logo_results = pickle.load(f)
    scores_lan = logo_results['scores_language']
    label_list = logo_results['labels']
    
    print("  Computing performance metrics...")
    rqueries_lan = []
    for i in range(len(set(groups_loci))):
        score_lan = scores_lan[i]
        y_test = label_list[i]
        try:
            roc_auc = roc_auc_score(y_test, score_lan)
            ranked_lan = [x for _, x in sorted(zip(score_lan, y_test), reverse=True)]
            rqueries_lan.append(ranked_lan)
        except:
            pass
    
    print("  Computing overall ROC AUC...")
    labels_flat = np.concatenate(label_list).ravel()
    scores_flat = np.concatenate(scores_lan).ravel()
    fpr, tpr, thrs = roc_curve(labels_flat, scores_flat)
    roc_auc = round(auc(fpr, tpr), 3)
    print(f"  Overall ROC AUC: {roc_auc}")
    
    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

