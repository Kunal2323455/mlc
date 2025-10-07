#!/usr/bin/env python3
"""
Quick Start Script for Tulsi AI System
Handles setup, training, and launching the complete system
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("🌿" + "="*60 + "🌿")
    print("    TULSI AI - Plant Disease Detection System")
    print("    Advanced Multi-Model AI with Web Interface")
    print("🌿" + "="*60 + "🌿")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import tensorflow
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import fastapi
        import uvicorn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("📂 Checking dataset...")
    
    if not os.path.exists('dataset.zip'):
        print("❌ dataset.zip not found")
        print("💡 Please ensure dataset.zip is in the current directory")
        return False
    
    if os.path.exists('dataset'):
        classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
        expected_classes = ['bacterial', 'fungal', 'healthy', 'pests']
        
        # Allow for fungi/fungal variation
        if 'fungi' in classes:
            classes = [cls if cls != 'fungi' else 'fungal' for cls in classes]
        
        missing = set(expected_classes) - set(classes)
        if missing:
            print(f"⚠️ Missing classes: {missing}")
        else:
            print(f"✅ Dataset ready with classes: {classes}")
            return True
    
    print("⚠️ Dataset not extracted yet (will be handled during training)")
    return True

def check_models():
    """Check if models are trained"""
    print("🧠 Checking AI models...")
    
    model_files = [
        "best_Custom_CNN_model.h5",
        "best_VGG16_Transfer_model.h5", 
        "best_MobileNet_Transfer_model.h5",
        "best_EfficientNet_Transfer_model.h5",
        "best_ResNet50_Transfer_model.h5",
        "model_config.json"
    ]
    
    existing = [f for f in model_files if os.path.exists(f)]
    
    if len(existing) >= 2:  # At least 2 models + config
        print(f"✅ Found {len(existing)} model files")
        return True
    else:
        print(f"⚠️ Only {len(existing)} model files found")
        print("💡 Models need to be trained")
        return False

def train_models():
    """Train the AI models"""
    print("\n🚀 Starting AI model training...")
    print("⏱️ This will take 30-60 minutes depending on your hardware")
    print("📊 Training 5 different AI models with ensemble learning")
    
    response = input("\n🤔 Do you want to proceed with training? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled")
        return False
    
    try:
        print("\n🔥 Training in progress...")
        print("💡 You can monitor progress in the terminal output")
        
        # Run main.py training
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Model training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return False

def start_api_server():
    """Start the API server"""
    print("\n🚀 Starting Tulsi AI server...")
    
    try:
        # Start the API server
        print("🌐 Server starting at http://localhost:8000")
        print("📱 Frontend will be available at the same URL")
        print("📚 API documentation at http://localhost:8000/docs")
        print("\n⏹️ Press Ctrl+C to stop the server")
        
        # Give user a moment to read
        time.sleep(2)
        
        # Open browser
        try:
            webbrowser.open('http://localhost:8000')
            print("🌐 Opening browser...")
        except:
            print("💡 Manually open: http://localhost:8000")
        
        # Start server
        subprocess.run([sys.executable, "api.py"])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False

def run_tests():
    """Run integration tests"""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_frontend_integration.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main function"""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first")
        return 1
    
    # Step 2: Check dataset
    if not check_dataset():
        print("\n❌ Please ensure dataset.zip is available")
        return 1
    
    # Step 3: Check models
    models_exist = check_models()
    
    # Step 4: Train models if needed
    if not models_exist:
        print("\n🎯 AI models need to be trained first")
        if not train_models():
            print("\n❌ Cannot proceed without trained models")
            return 1
    else:
        print("✅ AI models are ready")
    
    # Step 5: Ask what to do
    print("\n🎯 What would you like to do?")
    print("1. 🚀 Start Tulsi AI Web Interface")
    print("2. 🧪 Run System Tests")
    print("3. 🔄 Retrain Models")
    print("4. 📖 View Documentation")
    print("5. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            start_api_server()
            break
        elif choice == '2':
            if run_tests():
                print("\n✅ All tests passed!")
                print("💡 You can now start the web interface")
            else:
                print("\n❌ Some tests failed")
            break
        elif choice == '3':
            if train_models():
                print("\n✅ Retraining completed!")
            break
        elif choice == '4':
            print("\n📖 Documentation files:")
            print("   • README_IMPROVED.md - Complete project overview")
            print("   • FRONTEND_COMPLETE_GUIDE.md - Frontend documentation")
            print("   • MAIN_PY_ENHANCEMENTS.md - Model improvements")
            break
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)