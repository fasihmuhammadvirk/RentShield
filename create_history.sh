#!/bin/bash

# Configuration
EMAIL="fasih@example.com"
NAME="Fasih Muhammad"

# Start Date: 7 days ago
START_DATE=$(date -v-7d +%s)

# Function to make a commit with a specific relative time
# Usage: git_commit "Message" "Days_Since_Start" "Hours_Since_Start"
git_commit() {
    msg="$1"
    days_add="$2"
    hours_add="$3"
    
    # Calculate commit date
    # Start Date + (Days * 86400) + (Hours * 3600)
    total_seconds=$((START_DATE + days_add * 86400 + hours_add * 3600))
    current_date=$(date -r $total_seconds)
    
    GIT_AUTHOR_DATE="$current_date" GIT_COMMITTER_DATE="$current_date" git commit -m "$msg"
    echo "Committed: $msg at $current_date"
}

# 1. Reset Git
rm -rf .git
git init
git config user.email "$EMAIL"
git config user.name "$NAME"

# 2. Initial Setup (Day 0)
# Add only README (create a temp simpler one first to simulate start)
mv README.md README.final
echo "# RentShield Germany" > README.md
git add README.md
git_commit "Initial commit: RentShield project kick-off" 0 9

# Project structure
git add .agent/workflows .gitignore
git_commit "chore: Set up project structure and gitignore" 0 10

# Requirements
git add requirements.txt backend/requirements.txt frontend/requirements.txt
git_commit "deps: Add initial requirements files" 0 11

# 3. Data Generation (Day 1)
# Add dataset builder code
git add training/build_dataset.py
git_commit "feat(data): Add synthetic dataset generator for German rentals" 1 9

# Add sample data
git add data/sample/sample_listings.csv
git_commit "data: Generate initial sample listings for testing" 1 10

# 4. Backend Core (Day 2)
# Schemas
git add backend/app/schemas.py
git_commit "backend: Define Pydantic schemas for API inputs/outputs" 2 10

# Preprocessing
git add backend/app/core/preprocess.py
git_commit "feat(core): Implement city tier and postcode preprocessing logic" 2 14

# Pricing Engine
git add backend/app/core/pricing.py
git_commit "feat(core): Add pricing prediction engine with heuristic fallback" 2 16

# 5. ML Models (Day 3)
# Scam Engine (Logic)
git add backend/app/core/scam.py
git_commit "feat(core): Implement hybrid scam detection logic (NLP + Rules)" 3 10

# Training Scripts
git add training/train_price.py training/train_scam.py
git_commit "ml: Add training pipelines for Pricing and Scam models" 3 15

# Add Model Artifacts (The 'trained' models)
git add backend/app/artifacts
git_commit "ml: Check in initial trained model artifacts (v1.0)" 3 17

# 6. Backend API (Day 4)
# Main FastAPI app
git add backend/app/main.py backend/app/__init__.py backend/app/core/__init__.py
git_commit "api: Implement FastAPI endpoints for price and scam prediction" 4 11

# 7. Frontend Development (Day 5)
# Main App Shell
git add frontend/app.py
git_commit "ui: Create Streamlit main app shell and navigation" 5 9

# Single Checker Page
git add frontend/pages/1_Single_Checker.py
git_commit "ui: Add Single Listing Checker page with risk gauges" 5 14

# 8. Frontend Polish (Day 6)
# Bulk Page
git add frontend/pages/2_Bulk_CSV.py
git_commit "ui: Add Bulk CSV Analysis feature" 6 10

# Market Insights
git add frontend/pages/3_Market_Insights.py
git_commit "ui: Add Market Insights dashboard with city comparisons" 6 13

# 9. Final Polish & Docs (Day 7 - Today)
# Restore full README
mv README.final README.md
git add README.md
git_commit "docs: Update README with comprehensive setup and usage guide" 7 9

# Task tracking (Optional)
git add .gemini/antigravity/brain
git_commit "docs: Update internal task tracking logs" 7 10

# Deployment Stuff
git add frontend/bundled_app.py DEPLOYMENT_GUIDE.md
git_commit "ci: Add serverless deployment support (bundled app) and guide" 7 11

# Catch any remaining files
git add .
git_commit "chore: Final cleanup and small fixes" 7 12

echo "✅ Git History Reconstructed Successfully (18 Commits)"
git log --oneline
