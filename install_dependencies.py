#!/usr/bin/env python3
"""
Install script for tulsi disease detection dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Installing Tulsi Disease Detection Dependencies...")
    print("="*60)
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    # Read requirements
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📦 Found {len(requirements)} packages to install:")
    for req in requirements:
        print(f"   - {req}")
    
    print("\n🚀 Starting installation...")
    
    # Install all requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\n🔍 Trying individual package installation...")
        
        # Try installing packages individually
        failed_packages = []
        for req in requirements:
            print(f"\nInstalling {req}...")
            if install_package(req):
                print(f"✅ {req} installed")
            else:
                print(f"❌ {req} failed")
                failed_packages.append(req)
        
        if failed_packages:
            print(f"\n❌ Failed to install: {failed_packages}")
            print("\n💡 Try installing these manually:")
            for pkg in failed_packages:
                print(f"   pip install {pkg}")
            return False
        else:
            print("\n✅ All packages installed successfully!")
            return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Installation complete! You can now run:")
        print("   python3 main.py")
    else:
        print("\n⚠️ Installation incomplete. Please resolve the issues above.")
    
    sys.exit(0 if success else 1)