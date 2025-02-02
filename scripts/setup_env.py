#!/usr/bin/env python3
"""
Environment setup script to verify all required components are in place.
Checks for required environment variables and directory structure.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

def check_env_vars():
    """Verify all required environment variables are set."""
    required_vars = {
        "HF_TOKEN": "Hugging Face API token",
        "WANDB_API_KEY": "Weights & Biases API key",
        "WANDB_ENTITY": "Weights & Biases username",
        "PROJECT_ROOT": "Project root directory"
    }
    
    missing = []
    for var, desc in required_vars.items():
        if not os.getenv(var):
            missing.append(f"❌ {var}: {desc}")
    
    return missing

def check_hf_token(token):
    """Verify Hugging Face token is valid."""
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    print(f"🔑 Response: {r.status_code}")
    return r.status_code == 200

def check_directories():
    """Verify required directory structure exists."""
    required_dirs = [
        "configs/models",
        "models",
        "models/pretrained",
        "models/finetuned",
        "models/utils",
        "scripts"
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = Path(os.getenv("PROJECT_ROOT")) / dir_path
        if not full_path.exists():
            missing.append(f"❌ {dir_path}")
            # Create the directory if it doesn't exist
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    return missing

def main():
    """Run all environment checks."""
    print("🔍 Checking environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    missing_vars = check_env_vars()
    if missing_vars:
        print("\n⚠️ Missing environment variables:")
        for var in missing_vars:
            print(var)
    else:
        print("✅ All environment variables set")
    
    # Verify HF token
    # TODO: this checking code is buggy right now!
    # if os.getenv("HF_TOKEN"):
    #     if check_hf_token(os.getenv("HF_TOKEN")):
    #         print("✅ Hugging Face token verified")
    #     else:
    #         print("❌ Invalid Hugging Face token")
    
    # Check directory structure
    missing_dirs = check_directories()
    if missing_dirs:
        print("\n⚠️ Created missing directories:")
        for dir_path in missing_dirs:
            print(dir_path)
    else:
        print("✅ All required directories exist")
    
    # Overall status
    if not missing_vars:
        print("\n🎉 Environment setup complete!")
        return 0
    else:
        print("\n⚠️ Please fix the above issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
