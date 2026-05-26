import torch
import torch.optim as optim
import random
from utils import create_heterogeneous_data, split_data, evaluate_metrics, EarlyStopping
from asl import AsymmetricLoss
from modelHAN import HAN, MultiLabelToxicityClassifier, weighted_label_fusion, dynamic_contrastive_loss, GTN, FastGTN
import os
import numpy as np
import time
import dgl
import argparse
import json
import sys
import subprocess
import gc
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Force CPU execution to avoid DGL Windows/GPU instability
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # Hide GPU
# device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_gtn_data(g, device):
    # 1. Features
    h_feat = g.nodes['herb'].data['feat']
    i_feat = g.nodes['ingredient'].data['feat']
    t_feat = g.nodes['target'].data['feat']
    X = torch.cat([h_feat, i_feat, t_feat], dim=0).to(device)
    
    # 2. Adjacency
    num_nodes = X.shape[0]
    num_h = h_feat.shape[0]
    num_i = i_feat.shape[0]
    num_t = t_feat.shape[0]
    
    # Offsets
    off_h = 0
    off_i = num_h
    off_t = num_h + num_i
    
    # Edges
    # 0: H-I, 1: I-H
    src, dst = g.edges(etype='hi')
    hi_indices = torch.stack([src + off_h, dst + off_i])
    ih_indices = torch.stack([dst + off_i, src + off_h])
    
    # 2: H-T, 3: T-H
    src, dst = g.edges(etype='ht')
    ht_indices = torch.stack([src + off_h, dst + off_t])
    th_indices = torch.stack([dst + off_t, src + off_h])
    
    # 4: I-T, 5: T-I
    src, dst = g.edges(etype='it')
    it_indices = torch.stack([src + off_i, dst + off_t])
    ti_indices = torch.stack([dst + off_t, src + off_i])
    
    # Construct dense adjacency tensor (num_types, N, N)
    # 6 types + 1 identity
    A = torch.zeros(7, num_nodes, num_nodes, device=device)
    
    def fill_adj(indices, type_idx):
        if indices.shape[1] > 0:
            indices = indices.to(device)
            A[type_idx, indices[0], indices[1]] = 1.0
        
    fill_adj(hi_indices, 0)
    fill_adj(ih_indices, 1)
    fill_adj(ht_indices, 2)
    fill_adj(th_indices, 3)
    fill_adj(it_indices, 4)
    fill_adj(ti_indices, 5)
    
    # Add self-loops as type 6
    A[6] = torch.eye(num_nodes, device=device)
    
    return A, X

# Data Augmentation Functions
def drop_edge(A, drop_rate=0.1):
    if drop_rate <= 0:
        return A
    
    # A is (num_edge_types, N, N)
    # We drop edges independently for each type
    aug_A = A.clone()
    
    # Generate random mask
    mask = torch.rand_like(aug_A) > drop_rate
    
    # Apply mask (keep edges with prob 1-drop_rate)
    # Ensure diagonal (self-loops) are not dropped if they are in a specific channel?
    # Usually we don't drop self-loops.
    # In our GTN prep, A[6] is self-loops. Let's not drop those.
    
    # Only drop for first 6 types
    aug_A[:6] = aug_A[:6] * mask[:6]
    
    return aug_A

def feature_masking(X, drop_rate=0.1):
    if drop_rate <= 0:
        return X
    
    mask = torch.rand_like(X) > drop_rate
    return X * mask

def add_noise(X, noise_rate=0.1):
    if noise_rate <= 0:
        return X
    
    noise = torch.randn_like(X) * noise_rate
    return X + noise

def train_and_validate(g, gs, splits, num_epochs, fold_id_to_run=None, verbose=True, aug_type='none'):
    asl_loss = AsymmetricLoss()
    
    train_val_indices = splits["train_val_indices"]
    train_val_splits = splits["train_val_splits"]
    test_indices = splits["test_indices"]

    fold_results = []  # 存储每一折的评估结果

    # Move graphs to device
    g = g.to(device)
    # gs = [graph.to(device) for graph in gs] # Skipped for GTN
    labels = g.nodes['herb'].data['label']
    
    # Prepare GTN data
    A_orig, X_orig = prepare_gtn_data(g, device)

    # Determine which folds to run
    if fold_id_to_run is not None:
        folds_to_process = [(fold_id_to_run, train_val_splits[fold_id_to_run])]
    else:
        folds_to_process = list(enumerate(train_val_splits))

    for fold, (train_indices, val_indices) in folds_to_process:
        if verbose:
            print(f"Fold {fold + 1}/{len(train_val_splits)}")

        # Ensemble Loop Logic
        # If aug_type is 'ensemble', we run 3 models. Otherwise 1.
        num_models = 3 if aug_type == 'ensemble' else 1
        ensemble_preds_fusion_val = [] # Store VALIDATION predictions
        ensemble_preds_pure_val = []
        ensemble_preds_fusion_test = [] # Store TEST predictions
        ensemble_preds_pure_test = []

        val_target = None # To store validation targets
        test_target = None # To store test targets

        for model_idx in range(num_models):
            if num_models > 1 and verbose:
                print(f"  Training Model {model_idx + 1}/{num_models}...")
            
            # Re-seed for each model in ensemble to ensure diversity if needed
            # But wait, we want diversity? If we set seed same, initialization is same.
            # If we want ensemble, we should VARY the seed slightly or rely on different dropout/augmentation.
            # Augmentation is already random per epoch.
            # But initialization matters. Let's perturb seed for subsequent models?
            # Or just rely on random data augmentation.
            # Usually for ensemble we want different initializations.
            # Current setup_seed at start of function fixes everything.
            # We should probably set a different seed for each model if we want weight diversity.
            # Let's add model_idx to seed?
            # No, user wants reproducibility. So seed should be deterministic based on run_id/fold_id/model_idx.
            setup_seed(1000 + fold * 100 + model_idx) # Example deterministic variation

            # Initialize models for EACH fold to prevent data leakage/state carry-over
            embedding_model = FastGTN(
                num_edge_types=7,
                num_channels=2,
                in_dim=300,
                hidden_dim=128, # Reverted to 128 as per analysis
                out_dim=64, # 64*2 = 128
                num_layers=2
            ).to(device)
            
            classifier_model = MultiLabelToxicityClassifier(input_dim=128, hidden_dim=128, dropout_prob=0.1, output_dim=5).to(device)
            
            optimizer = optim.Adam(
                list(embedding_model.parameters()) + list(classifier_model.parameters()),
                lr=0.001, weight_decay=1e-5
            )

            # Create masks
            train_mask = torch.zeros(g.num_nodes('herb'), dtype=torch.bool).to(device)
            val_mask = torch.zeros(g.num_nodes('herb'), dtype=torch.bool).to(device)
            test_mask = torch.zeros(g.num_nodes('herb'), dtype=torch.bool).to(device)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True
            
            stopper = EarlyStopping(patience=10)

            for epoch in range(num_epochs):
                embedding_model.train()
                classifier_model.train()
                optimizer.zero_grad()
                
                # Apply Augmentation
                curr_A = A_orig
                curr_X = X_orig
                
                # Augmentation logic
                # Baseline uses feature_mask by default if aug_type is 'baseline' or 'feature_mask'
                use_aug = True
                if aug_type == 'baseline' or aug_type == 'feature_mask':
                    curr_X = feature_masking(X_orig, drop_rate=0.2)
                elif aug_type == 'ensemble':
                    # These strategies also use feature_mask as base
                    curr_X = feature_masking(X_orig, drop_rate=0.2)
                else:
                    use_aug = False

                # Forward pass
                embeddings_all = embedding_model(curr_A, curr_X)
                embeddings = embeddings_all[:g.num_nodes('herb')]
                
                out = classifier_model(embeddings)
                
                # Loss Calculation based on Strategy
                targets = labels[train_mask]
                
                # Default ASL (Ensemble mode usually doesn't need smooth label, 
                # but we can add it if you want. Let's stick to standard ASL first for consistency)
                clf_loss = asl_loss(out[train_mask], targets)
                
                contrastive_loss_val = dynamic_contrastive_loss(embeddings[train_mask], targets)

                lam = 0.1
                total_loss = clf_loss + lam * contrastive_loss_val

                total_loss.backward()
                optimizer.step()

                # Validation and Early Stopping
                if (epoch + 1) % 20 == 0:
                    embedding_model.eval()
                    classifier_model.eval()
                    with torch.no_grad():
                        # No augmentation during validation/inference
                        embeddings_all = embedding_model(A_orig, X_orig)
                        embeddings = embeddings_all[:g.num_nodes('herb')]
                        
                        train_embs = embeddings[train_mask]
                        train_labels_batch = labels[train_mask]
                        val_embs = embeddings[val_mask]
                        val_target = labels[val_mask]

                        val_out1 = classifier_model(embeddings)[val_mask]
                        
                        predictions = []
                        for val_emb in val_embs:
                            fused_label = weighted_label_fusion(val_emb, train_embs, train_labels_batch)
                            predictions.append(fused_label)
                        val_out2 = torch.stack(predictions)

                        alpha = 0.6
                        final_val_out = alpha * val_out1 + (1 - alpha) * val_out2

                        val_loss = asl_loss(final_val_out, val_target)
                        
                        if verbose:
                            print(f"Epoch {epoch + 1}, Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                        
                        stop = stopper.step(val_loss.item(), {
                            'embedding': embedding_model.state_dict(), 
                            'classifier': classifier_model.state_dict()
                        })
                        
                        if stop:
                            # if verbose:
                            #     print(f"Early stopping at epoch {epoch + 1}")
                            break

            # Final Evaluation on TEST set using BEST model for this member
            checkpoint = stopper.load_checkpoint()
            embedding_model.load_state_dict(checkpoint['embedding'])
            classifier_model.load_state_dict(checkpoint['classifier'])
            
            embedding_model.eval()
            classifier_model.eval()
            with torch.no_grad():
                embeddings_all = embedding_model(A_orig, X_orig)
                embeddings = embeddings_all[:g.num_nodes('herb')]
                
                train_embs = embeddings[train_mask]
                train_labels_batch = labels[train_mask]
                
                # --- Validation Phase (Store for Ensemble) ---
                val_embs = embeddings[val_mask]
                val_target = labels[val_mask] # Update global var (same for all models)
                val_out1 = classifier_model(embeddings)[val_mask]
                
                val_predictions = []
                for val_emb in val_embs:
                    fused_label = weighted_label_fusion(val_emb, train_embs, train_labels_batch)
                    val_predictions.append(fused_label)
                val_out2 = torch.stack(val_predictions)
                
                alpha = 0.6
                final_val_out = alpha * val_out1 + (1 - alpha) * val_out2
                
                ensemble_preds_fusion_val.append(final_val_out)
                ensemble_preds_pure_val.append(val_out1)

                # --- Test Phase (Store for Ensemble) ---
                test_embs = embeddings[test_mask]
                test_target = labels[test_mask] # Update global var

                test_out1 = classifier_model(embeddings)[test_mask]
                
                predictions = []
                for test_emb in test_embs:
                    fused_label = weighted_label_fusion(test_emb, train_embs, train_labels_batch)
                    predictions.append(fused_label)
                test_out2 = torch.stack(predictions)

                final_test_out = alpha * test_out1 + (1 - alpha) * test_out2
                
                ensemble_preds_fusion_test.append(final_test_out)
                ensemble_preds_pure_test.append(test_out1)
            
            # Cleanup memory
            del embedding_model, classifier_model, optimizer
            gc.collect()

        # Average Predictions if Ensemble
        final_val_out_avg = torch.stack(ensemble_preds_fusion_val).mean(dim=0)
        final_val_pure_avg = torch.stack(ensemble_preds_pure_val).mean(dim=0)
        
        final_test_out_avg = torch.stack(ensemble_preds_fusion_test).mean(dim=0)
        final_test_pure_avg = torch.stack(ensemble_preds_pure_test).mean(dim=0)

        # Determine best thresholds on VALIDATION set (using Averaged Predictions)
        val_metrics = evaluate_metrics(val_target, final_val_out_avg, threshold=None)
        best_val_threshold = val_metrics["Best Threshold"]
        
        val_metrics_pure = evaluate_metrics(val_target, final_val_pure_avg, threshold=None)
        best_val_threshold_pure = val_metrics_pure["Best Threshold"]
        
        if verbose:
            print(f"Best Validation Threshold (Fusion): {best_val_threshold:.4f}")
            print(f"Best Validation Threshold (Pure): {best_val_threshold_pure:.4f}")

        # Evaluate on TEST set using Best Validation Threshold
        metrics = evaluate_metrics(test_target, final_test_out_avg, threshold=best_val_threshold)
        metrics["Type"] = "With Fusion"
        fold_results.append(metrics)

        metrics_pure = evaluate_metrics(test_target, final_test_pure_avg, threshold=best_val_threshold_pure)
        metrics_pure["Type"] = "Without Fusion (Pure MLP)"
        fold_results.append(metrics_pure)

    return fold_results

def calculate_mean_std(fold_results):
    metrics_keys = [k for k in fold_results[0].keys() if k != "Type"] # Exclude non-numeric 'Type' field
    mean_std_results = {}
    for key in metrics_keys:
        values = [result[key] for result in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        mean_std_results[key] = f"{mean:.4f} ± {std:.4f}"
    return mean_std_results

def run_single_experiment(run_id, fold_id, num_epochs, num_folds=5, test_ratio=0.15, aug_type='none'):
    print(f"\n========== Run {run_id + 1} Fold {fold_id + 1} [Aug: {aug_type}] ==========")
    setup_seed(1000 + run_id + fold_id)
    g = create_heterogeneous_data()
    print("Generating meta-path graphs... (Skipped for GTN)")
    # g_hih = dgl.metapath_reachable_graph(g, ['hi', 'ih'])
    # g_hth = dgl.metapath_reachable_graph(g, ['ht', 'th'])
    # gs = [g_hih, g_hth]
    gs = [] # Placeholder
    
    splits = split_data(g, test_ratio=test_ratio, num_folds=num_folds, random_seed=1000 + run_id)
    fold_results = train_and_validate(g, gs, splits, num_epochs, fold_id_to_run=fold_id, verbose=True, aug_type=aug_type)
    return fold_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=-1, help="Run ID for single process execution")
    parser.add_argument("--fold_id", type=int, default=-1, help="Fold ID for single process execution")
    parser.add_argument("--aug_type", type=str, default='none', help="Augmentation type: none, drop_edge, feature_mask, noise")
    args = parser.parse_args()

    num_runs = 10
    num_folds = 5
    test_ratio = 0.15
    num_epochs = 200
    
    # Comparison Mode (if parent process)
    # aug_strategies = ['none', 'drop_edge', 'feature_mask', 'noise']

    # Child Process Mode
    if args.run_id != -1 and args.fold_id != -1:
        try:
            # Force Ensemble + Feature Masking (Best Strategy)
            # We treat 'ensemble' as the aug_type to trigger the loop in train_and_validate
            aug_type = 'ensemble' 
            results = run_single_experiment(args.run_id, args.fold_id, num_epochs, num_folds, test_ratio, aug_type)
            # Save results to a temporary JSON file
            with open(f"temp_results_run_{args.run_id}_fold_{args.fold_id}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            print(f"Run {args.run_id} Fold {args.fold_id} failed with error: {e}")
            sys.exit(1)
        return

    # Parent Process Mode (Orchestrator)
    print("Starting Final Best Strategy (Ensemble + Feature Masking) - 10 runs x 5 folds...")
    
    all_fold_results = []
    
    for i in range(num_runs):
        print(f"\n========== Dispatching Run {i+1}/{num_runs} ==========")
        for j in range(num_folds):
            print(f"Running Fold {j+1}/{num_folds}...")
            
            # Launch subprocess for this fold
            # Passing 'ensemble' to trigger the correct logic in child process
            cmd = [sys.executable, __file__, "--run_id", str(i), "--fold_id", str(j), "--aug_type", "ensemble"]
            process = subprocess.Popen(cmd)
            process.wait()
            
            # Read results back
            result_file = f"temp_results_run_{i}_fold_{j}.json"
            if process.returncode == 0 and os.path.exists(result_file):
                with open(result_file, "r") as f:
                    fold_result = json.load(f)
                    all_fold_results.extend(fold_result) # fold_result is a list of 1 dict
                    print(f"Run {i+1} Fold {j+1} Completed: {fold_result[0]}")
                os.remove(result_file)
            else:
                print(f"Error: Run {i+1} Fold {j+1} failed.")

    if all_fold_results:
        print(f"\n================ Final {num_runs}x{num_folds} CV Results (Ensemble) ================")
        
        # Separate results by Type
        fusion_results = [r for r in all_fold_results if r.get("Type") == "With Fusion"]
        pure_results = [r for r in all_fold_results if r.get("Type") == "Without Fusion (Pure MLP)"]
        
        if fusion_results:
            print("\n--- With Fusion Results ---")
            mean_std_results = calculate_mean_std(fusion_results)
            for key, value in mean_std_results.items():
                print(f"{key}: {value}")
                
        if pure_results:
            print("\n--- Without Fusion (Pure MLP) Results ---")
            mean_std_results = calculate_mean_std(pure_results)
            for key, value in mean_std_results.items():
                print(f"{key}: {value}")
    else:
        print("No successful runs completed.")

if __name__ == "__main__":
    begin_time = time.time()
    main()
    if "--run_id" not in sys.argv:
        print(f"Total time: {time.time() - begin_time:.2f} seconds.")
