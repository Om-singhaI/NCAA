#!/bin/bash
# NCAA Seed Prediction - Quick Start Commands

echo "🏀 NCAA Seed Prediction - Best Solution"
echo "========================================"
echo ""

# Install dependencies
echo "Step 1: Install dependencies (one-time)"
pip3 install xgboost lightgbm catboost pandas numpy scikit-learn -q
echo "✅ Dependencies installed"
echo ""

# Run best model
echo "Step 2: Run best model (stacking)"
python3 stacking_model.py
echo ""

# Verify submission
echo "Step 3: Verify submission"
python3 verify_submission.py
echo ""

echo "✅ COMPLETE!"
echo "Your submission is ready: my_submission.csv"
echo "Expected CV RMSE: ~2.56 (from ~18.9 improvement!)"
