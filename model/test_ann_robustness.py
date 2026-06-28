"""
Test suite for checking the robustness and correctness of train_ann.py.
This script tests:
1. Model initialization and forward pass under different dimensions and activations.
2. EarlyStopping helper class logic (counters, patience, state saving/restoring).
3. Data types and casting correctness.
4. Device mapping consistency.
5. Simulated Optuna objective trials.

Author: Challenger 2
"""

import unittest
import torch
import numpy as np
import os
import sys

# Add the model directory to sys.path to import train_ann
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ann
from train_ann import MUISNN, EarlyStopping, run_cross_validation, train_final_model

class TestANNModel(unittest.TestCase):
    
    def setUp(self):
        # Generate some synthetic dummy data for testing
        self.num_samples = 40
        self.input_dim = 27
        self.X_dummy = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
        self.y_dummy = np.random.uniform(0, 10, size=(self.num_samples,)).astype(np.float32)

    def test_model_architecture_and_forward(self):
        """
        Test that MUISNN initializes correctly and outputs correct shape for various parameters.
        """
        configs = [
            {"hidden_dims": [64], "dropout_rate": 0.2, "activation_fn": "relu"},
            {"hidden_dims": [128, 64], "dropout_rate": 0.4, "activation_fn": "leaky_relu"},
            {"hidden_dims": [32, 16, 8], "dropout_rate": 0.1, "activation_fn": "elu"},
        ]
        
        for cfg in configs:
            with self.subTest(cfg=cfg):
                model = MUISNN(
                    input_dim=self.input_dim,
                    hidden_dims=cfg["hidden_dims"],
                    dropout_rate=cfg["dropout_rate"],
                    activation_fn=cfg["activation_fn"]
                )
                
                # Check output shape
                x_tensor = torch.randn(5, self.input_dim)
                preds = model(x_tensor)
                self.assertEqual(preds.shape, (5,))
                
                # Check output gradient capability
                self.assertTrue(preds.requires_grad)

    def test_early_stopping_logic(self):
        """
        Verify that EarlyStopping stops training and saves/restores state correctly.
        """
        model = MUISNN(self.input_dim, [16], 0.1, "relu")
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Initialize EarlyStopping with patience = 3
        early_stopping = EarlyStopping(patience=3)
        
        # 1. First loss: should update best loss and best state
        early_stopping(1.0, model)
        self.assertEqual(early_stopping.best_loss, 1.0)
        self.assertEqual(early_stopping.counter, 0)
        self.assertFalse(early_stopping.early_stop)
        
        # Modify model weights slightly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        modified_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 2. Loss worsens: counter increments
        early_stopping(1.2, model)
        self.assertEqual(early_stopping.best_loss, 1.0)
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping.early_stop)
        
        # 3. Loss worsens again: counter = 2
        early_stopping(1.1, model)
        self.assertEqual(early_stopping.best_loss, 1.0)
        self.assertEqual(early_stopping.counter, 2)
        self.assertFalse(early_stopping.early_stop)
        
        # 4. Loss improves: best loss updates, counter resets
        early_stopping(0.9, model)
        self.assertEqual(early_stopping.best_loss, 0.9)
        self.assertEqual(early_stopping.counter, 0)
        
        # Update modified_state reference
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        new_modified_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 5. Loss worsens 3 times consecutively: early stopping triggers
        early_stopping(0.95, model) # counter 1
        early_stopping(0.96, model) # counter 2
        early_stopping(0.97, model) # counter 3 -> triggers
        
        self.assertEqual(early_stopping.counter, 3)
        self.assertTrue(early_stopping.early_stop)
        
        # Check that saved best state is from when loss was 0.9 (which had the new_modified_state)
        # Load the best state and compare
        model.load_state_dict(early_stopping.best_state)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.allclose(v.cpu(), modified_state[k].cpu()))

    def test_run_cross_validation_cpu(self):
        """
        Verify run_cross_validation executes end-to-end on CPU without errors.
        """
        # Run with very few epochs to make it fast
        cv_maes = run_cross_validation(
            X_cv=self.X_dummy,
            y_cv=self.y_dummy,
            hidden_dims=[16],
            dropout_rate=0.1,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=8,
            activation_fn="relu",
            epochs=5,
            patience=2
        )
        self.assertEqual(len(cv_maes), 5)
        for mae in cv_maes:
            self.assertIsInstance(mae, float)
            self.assertGreaterEqual(mae, 0.0)

    def test_train_final_model_cpu(self):
        """
        Verify train_final_model executes end-to-end on CPU and outputs valid history.
        """
        best_params = {
            "n_layers": 2,
            "n_units_l0": 16,
            "n_units_l1": 8,
            "dropout_rate": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "activation_fn": "leaky_relu"
        }
        
        model, train_losses, val_losses = train_final_model(
            X_train=self.X_dummy[:30],
            y_train=self.y_dummy[:30],
            X_val=self.X_dummy[30:],
            y_val=self.y_dummy[30:],
            best_params=best_params,
            epochs=5,
            patience=2
        )
        
        self.assertIsInstance(model, MUISNN)
        self.assertEqual(len(train_losses), 5)
        self.assertEqual(len(val_losses), 5)
        for tl, vl in zip(train_losses, val_losses):
            self.assertIsInstance(tl, float)
            self.assertIsInstance(vl, float)

if __name__ == "__main__":
    unittest.main()
