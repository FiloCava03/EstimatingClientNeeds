import torch
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
from NN.model import multi_task_loss

def train_multitask_model(model, train_loader, val_loader, optimizer, 
                          w_a, w_i, joint_prior, epochs=100, patience=15, lam=0.1):
    
    best_val_metric = 0 # Now tracking Mean PR-AUC instead of just ROC-AUC
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        
        for X_batch, y_inc, y_acc in train_loader:
            optimizer.zero_grad()
            logits_acc, logits_inc = model(X_batch)
            
            loss = multi_task_loss(
                logits_acc, logits_inc, 
                y_acc, y_inc, 
                w_a, w_i, joint_prior, lam
            )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        val_preds_acc, val_preds_inc = [], []
        val_targs_acc, val_targs_inc = [], []
        
        with torch.no_grad():
            for X_batch, y_inc, y_acc in val_loader:
                logits_acc, logits_inc = model(X_batch)
                
                val_preds_acc.extend(torch.sigmoid(logits_acc).numpy())
                val_preds_inc.extend(torch.sigmoid(logits_inc).numpy())
                val_targs_acc.extend(y_acc.numpy())
                val_targs_inc.extend(y_inc.numpy())
                
        # Calculate Validation Metrics
        auc_acc = roc_auc_score(val_targs_acc, val_preds_acc)
        auc_inc = roc_auc_score(val_targs_inc, val_preds_inc)
        
        # Calculate PR-AUC (crucial for imbalanced targets like Income)
        pr_acc = average_precision_score(val_targs_acc, val_preds_acc)
        pr_inc = average_precision_score(val_targs_inc, val_preds_inc)
        mean_pr = (pr_acc + pr_inc) / 2
        
        # --- EARLY STOPPING LOGIC ---
        # Now optimizing for the harder metric (PR-AUC)
        if mean_pr > best_val_metric:
            best_val_metric = mean_pr
            patience_counter = 0
            # FIX: Deepcopy prevents the "frozen" weights from overwriting themselves
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"ROC-AUC (Acc/Inc): {auc_acc:.3f}/{auc_inc:.3f} | PR-AUC (Acc/Inc): {pr_acc:.3f}/{pr_inc:.3f}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Restoring best weights.")
            break
            
    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_metric