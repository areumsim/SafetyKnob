#!/usr/bin/env python
"""
SafetyKnob - Industrial Image Safety Assessment System
Unified entry point with enhanced debugging support
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging(debug=False, verbose=False):
    """Configure logging with debug support"""
    log_level = logging.WARNING
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main entry point with debugging enhancements"""
    parser = argparse.ArgumentParser(
        description='SafetyKnob - Industrial Image Safety Assessment System',
        epilog='For more information, see: https://github.com/yourusername/safetyknob'
    )
    
    # Global options
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling mode')
    
    # Parse only known args to get global options
    args, remaining = parser.parse_known_args()
    
    # Setup logging based on global options
    setup_logging(args.debug, args.verbose)
    
    # Enable debugging features
    if args.debug:
        os.environ['SAFETYKNOB_DEBUG'] = '1'
        os.environ['SAFETYKNOB_LOG_LEVEL'] = 'DEBUG'
        # Enable Python development mode for better error messages
        if sys.version_info >= (3, 7):
            sys.flags.dev_mode = True
    
    # Import main module after setting up environment
    try:
        from main import main as app_main
    except ImportError as e:
        logging.error(f"Failed to import main module: {e}")
        logging.error("Make sure you're in the project root directory")
        sys.exit(1)
    
    # Run with profiling if requested
    if args.profile:
        import cProfile
        import pstats
        from datetime import datetime
        
        profile_file = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stats"
        logging.info(f"Profiling enabled. Results will be saved to: {profile_file}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Pass remaining arguments to main app
            sys.argv = [sys.argv[0]] + remaining
            app_main()
        finally:
            profiler.disable()
            profiler.dump_stats(profile_file)
            
            # Print top 20 time-consuming functions
            stats = pstats.Stats(profiler)
            print("\n=== Performance Profile ===")
            stats.sort_stats('cumulative').print_stats(20)
            print(f"\nFull profile saved to: {profile_file}")
    else:
        # Normal execution
        try:
            # Pass remaining arguments to main app
            sys.argv = [sys.argv[0]] + remaining
            app_main()
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            if args.debug:
                # In debug mode, show full traceback
                logging.exception("An error occurred:")
            else:
                # In normal mode, show cleaner error
                logging.error(f"Error: {e}")
                logging.error("Run with --debug for full traceback")
            sys.exit(1)

if __name__ == '__main__':
    main()