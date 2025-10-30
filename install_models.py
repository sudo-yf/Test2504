"""
Interactive installer for emotion recognition models.
Helps users choose and install the appropriate models.
"""

import subprocess
import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def install_package(package_name):
    """Install a package using pip."""
    print(f"\n📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False


def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    """Main installation wizard."""
    print_header("EmotiSense - Model Installation Wizard")
    
    print("""
This wizard will help you install emotion recognition models.

Available models:
1. HSEmotion (Recommended) - Fast and accurate
2. FER - Lightweight alternative
3. All models - Install everything
4. Skip - Use default DeepFace only

""")
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            # Install HSEmotion
            print_header("Installing HSEmotion")
            print("""
HSEmotion is a high-speed emotion recognition library with:
- 66%+ accuracy on AffectNet
- ~60ms inference time
- Multiple pre-trained models
            """)
            
            packages = ['hsemotion', 'timm>=0.9.0']
            success = all(install_package(pkg) for pkg in packages)
            
            if success:
                print("\n✅ HSEmotion installation complete!")
                print("\nTo use HSEmotion, set in config.yaml:")
                print("  emotion:")
                print("    detector_type: 'hsemotion'")
            break
            
        elif choice == '2':
            # Install FER
            print_header("Installing FER")
            print("""
FER is a lightweight facial expression recognition library with:
- ~65% accuracy on FER2013
- Simple API
- Real-time video support
            """)
            
            success = install_package('fer')
            
            if success:
                print("\n✅ FER installation complete!")
                print("\nTo use FER, set in config.yaml:")
                print("  emotion:")
                print("    detector_type: 'fer'")
            break
            
        elif choice == '3':
            # Install all
            print_header("Installing All Models")
            
            packages = ['hsemotion', 'timm>=0.9.0', 'fer']
            success = all(install_package(pkg) for pkg in packages)
            
            if success:
                print("\n✅ All models installed successfully!")
                print("\nYou can now use any detector in config.yaml:")
                print("  - 'hsemotion' (recommended)")
                print("  - 'fer'")
                print("  - 'deepface' (default)")
                print("  - 'ensemble' (combines multiple models)")
            break
            
        elif choice == '4':
            # Skip
            print("\n⏭️  Skipping model installation")
            print("You will use the default DeepFace detector")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")
    
    # Check installation
    print_header("Verifying Installation")
    
    models_status = {
        'DeepFace': check_package('deepface'),
        'HSEmotion': check_package('hsemotion'),
        'FER': check_package('fer'),
    }
    
    print("\nInstalled models:")
    for model, installed in models_status.items():
        status = "✅" if installed else "❌"
        print(f"  {status} {model}")
    
    # Final instructions
    print_header("Next Steps")
    print("""
1. Edit config.yaml to select your preferred detector:
   
   emotion:
     detector_type: 'hsemotion'  # or 'fer', 'deepface', 'ensemble'

2. Run the application:
   
   python main.py

3. (Optional) Compare models:
   
   python compare_models.py --mode webcam

For more information, see ADVANCED_MODELS.md
    """)


if __name__ == '__main__':
    main()

