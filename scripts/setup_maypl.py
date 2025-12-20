"""
Setup script for MAYPL integration.

This script helps set up the MAYPL repository for use with the kg_embedding module.
"""

import subprocess
import sys
from pathlib import Path


def setup_maypl(maypl_dir: str = "maypl", clone_url: str = "https://github.com/bdi-lab/MAYPL.git"):
    """
    Set up MAYPL repository.
    
    Args:
        maypl_dir: Directory name for MAYPL repository.
        clone_url: URL to clone MAYPL from.
    """
    print("=" * 60)
    print("MAYPL Setup Script")
    print("=" * 60)
    
    maypl_path = Path(maypl_dir)
    
    # Check if already exists
    if maypl_path.exists():
        print(f"✓ MAYPL directory already exists: {maypl_path}")
        print(f"  Set MAYPL_PATH environment variable to: {maypl_path.absolute()}")
        return str(maypl_path.absolute())
    
    # Clone repository
    print(f"\nCloning MAYPL repository from {clone_url}...")
    try:
        subprocess.run(
            ["git", "clone", clone_url, str(maypl_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ Successfully cloned MAYPL to {maypl_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error cloning repository: {e}")
        print("  Please clone manually: git clone https://github.com/bdi-lab/MAYPL.git")
        return None
    except FileNotFoundError:
        print("✗ Git not found. Please install git and try again.")
        print("  Or clone manually: git clone https://github.com/bdi-lab/MAYPL.git")
        return None
    
    # Install requirements
    requirements_file = maypl_path / "requirements.txt"
    if requirements_file.exists():
        print("\nInstalling MAYPL requirements...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            print("✓ Requirements installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Failed to install some requirements: {e}")
            print("  Please install manually: pip install -r maypl/requirements.txt")
    else:
        print("⚠ Warning: requirements.txt not found in MAYPL directory")
    
    # Set environment variable
    abs_path = maypl_path.absolute()
    print("\n✓ MAYPL setup complete!")
    print("\nTo use MAYPL, set the environment variable:")
    print(f"  export MAYPL_PATH={abs_path}")
    print("\nOr in Python:")
    print("  import os")
    print(f"  os.environ['MAYPL_PATH'] = '{abs_path}'")
    
    return str(abs_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup MAYPL for knowledge graph embedding")
    parser.add_argument(
        "--dir",
        default="maypl",
        help="Directory to clone MAYPL into (default: maypl)"
    )
    parser.add_argument(
        "--url",
        default="https://github.com/bdi-lab/MAYPL.git",
        help="MAYPL repository URL"
    )
    
    args = parser.parse_args()
    
    setup_maypl(args.dir, args.url)

