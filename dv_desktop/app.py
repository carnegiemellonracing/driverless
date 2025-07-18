#!/usr/bin/env python3
"""
DV Desktop - Main Entry Point
"""

import sys
import tkinter as tk
import tkinter.simpledialog
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.main_window import DVDesktop
from config.settings import AppSettings

def setup_application():
    """Initialize application settings and environment"""
    # Ensure required directories exist
    required_dirs = [
        PROJECT_ROOT / "assets" / "icons",
        PROJECT_ROOT / "assets" / "images", 
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "logs"
    ]
    
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = PROJECT_ROOT / "logs" / "dv_desktop.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main application entry point"""
    # Setup application environment
    logger = setup_application()
    logger.info("Starting DV Desktop Application")
    
    try:
        # Create main window
        root = tk.Tk()
        
        # Add simpledialog to tk namespace for folder creation
        tk.simpledialog = tkinter.simpledialog
        
        # Load application settings
        settings = AppSettings()
        
        # Initialize main application
        app = DVDesktop(root, settings)
        
        # Start main loop
        logger.info("Application initialized successfully")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error(traceback.format_exc())
        raise
    
    finally:
        logger.info("DV Desktop Application shutting down")

if __name__ == "__main__":
    main()