#!/usr/bin/env python3
"""
Main entry point for Industrial Image Safety Assessment System
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.safety_assessment_system import SafetyAssessmentSystem
from src.analysis.model_comparison import ModelPerformanceAnalyzer
from src.config.settings import SystemConfig

# Configure logging - will be reconfigured based on command line args
logger = logging.getLogger(__name__)


def configure_logging(debug=False, verbose=False):
    """Configure logging based on command line arguments"""
    # Get log level from environment or command line
    env_log_level = os.environ.get('SAFETYKNOB_LOG_LEVEL', '').upper()
    
    if debug or os.environ.get('SAFETYKNOB_DEBUG', ''):
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    elif env_log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        log_level = getattr(logging, env_log_level)
    else:
        log_level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Reconfigure even if already configured
    )
    
    # Adjust third-party library logging
    if not debug:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Image Safety Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assess a single image
  python main.py assess image.jpg
  
  # Train the system
  python main.py train --data-dir ./data/train
  
  # Evaluate on test set
  python main.py evaluate --data-dir ./data/test
  
  # Compare models
  python main.py compare --data-dir ./data/test
  
  # Start API server
  python main.py serve --port 8000
  
  # Run full experiment
  python main.py experiment --train-dir ./data/train --test-dir ./data/test
        """
    )
    
    # Global arguments
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Assess command
    assess_parser = subparsers.add_parser('assess', help='Assess safety of images')
    assess_parser.add_argument('path', type=str, help='Path to image file or directory')
    assess_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    assess_parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    assess_parser.add_argument('--output', type=str, help='Output file for batch results')
    assess_parser.add_argument('--pattern', type=str, default='*.jpg', help='File pattern for directory search')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the safety assessment system')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--labels', type=str, help='Labels JSON file')
    train_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate system performance')
    eval_parser.add_argument('--data-dir', type=str, required=True, help='Test data directory')
    eval_parser.add_argument('--labels', type=str, help='Labels JSON file')
    eval_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    eval_parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model performances')
    compare_parser.add_argument('--data-dir', type=str, required=True, help='Test data directory')
    compare_parser.add_argument('--labels', type=str, help='Labels JSON file')
    compare_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    compare_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    compare_parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    # Experiment command (replaces shell script functionality)
    experiment_parser = subparsers.add_parser('experiment', help='Run full experiment')
    experiment_parser.add_argument('--train-dir', type=str, required=True, help='Training data directory')
    experiment_parser.add_argument('--test-dir', type=str, required=True, help='Test data directory')
    experiment_parser.add_argument('--labels', type=str, help='Labels JSON file')
    experiment_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    experiment_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port number')
    serve_parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    serve_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging based on command line arguments
    configure_logging(args.debug, args.verbose)
    
    # Log debug information if debug mode is enabled
    if args.debug:
        logger.debug(f"Debug mode enabled")
        logger.debug(f"Command: {args.command}")
        logger.debug(f"Arguments: {vars(args)}")
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SystemConfig.from_dict(config_dict)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = SystemConfig()
    
    # Initialize system
    system = SafetyAssessmentSystem(config)
    
    if args.command == 'assess':
        # Check if path is file or directory
        path = Path(args.path)
        
        if path.is_file():
            # Single file assessment
            logger.info(f"Assessing image: {args.path}")
            result = system.assess_image(args.path)
            
            print(f"\n{'='*60}")
            print(f"Safety Assessment Results")
            print(f"{'='*60}")
            print(f"Image: {result.image_path}")
            print(f"Overall Safety: {'SAFE' if result.is_safe else 'UNSAFE'}")
            print(f"Safety Score: {result.overall_safety_score:.2%}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"\nRisk Summary: {result.get_risk_summary()}")
            
            if args.verbose:
                print(f"\nDetailed Dimension Scores:")
                for dim, score in result.dimension_scores.items():
                    risk_level = "Safe" if score > 0.5 else "Risk"
                    print(f"  - {dim.replace('_', ' ').title()}: {score:.2%} ({risk_level})")
                print(f"\nProcessing Time: {result.processing_time:.3f}s")
                print(f"Method Used: {result.method_used}")
                print(f"Model: {result.model_name}")
        
        elif path.is_dir():
            # Directory batch assessment
            from test_batch import test_batch_images
            
            logger.info(f"Processing directory: {args.path}")
            
            # Collect image files
            image_files = []
            patterns = args.pattern.split(',')
            
            if args.recursive:
                for pattern in patterns:
                    image_files.extend(str(p) for p in path.rglob(pattern))
            else:
                for pattern in patterns:
                    image_files.extend(str(p) for p in path.glob(pattern))
            
            if not image_files:
                print(f"No images found in {path} with pattern: {args.pattern}")
                return
            
            print(f"Found {len(image_files)} images to process")
            
            # Process batch
            output_file = args.output or f"batch_results_{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            test_batch_images(image_files, output_file)
        
        else:
            print(f"Error: Path not found: {args.path}")
    
    elif args.command == 'train':
        # Train system
        logger.info(f"Training on data from: {args.data_dir}")
        
        # Override config if epochs specified
        if args.epochs:
            config.training.epochs = args.epochs
        
        # Import here to avoid circular imports
        from src.utils import ImageDataset
        
        dataset = ImageDataset(
            args.data_dir,
            Path(args.labels) if args.labels else None
        )
        
        system.train(dataset, config.training)
        system.save_models()  # Save after training
        logger.info("Training completed successfully")
    
    elif args.command == 'evaluate':
        # Evaluate system
        logger.info(f"Evaluating on data from: {args.data_dir}")
        
        # Import here to avoid circular imports
        from src.utils import ImageDataset
        
        # Load models if available
        checkpoint_dir = Path(config.checkpoint_dir)
        if checkpoint_dir.exists():
            logger.info("Loading trained models...")
            system.load_models()
        
        dataset = ImageDataset(
            args.data_dir,
            Path(args.labels) if args.labels else None
        )
        
        results = system.evaluate_dataset(dataset)
        
        # Handle different result formats
        if 'ensemble_metrics' in results:
            metrics = results['ensemble_metrics']
        else:
            metrics = results
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['confusion_matrix']['tp']}")
        print(f"  TN: {metrics['confusion_matrix']['tn']}")
        print(f"  FP: {metrics['confusion_matrix']['fp']}")
        print(f"  FN: {metrics['confusion_matrix']['fn']}")
        
        # Show individual model results if available
        if 'individual_metrics' in results:
            print(f"\nIndividual Model Performance:")
            for model_name, model_metrics in results['individual_metrics'].items():
                print(f"  {model_name}: {model_metrics['f1_score']:.2%} F1 Score")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    elif args.command == 'compare':
        # Compare model performances
        logger.info(f"Comparing models on data from: {args.data_dir}")
        
        # Import here to avoid circular imports
        from src.utils import ImageDataset
        
        # Load models if available
        checkpoint_dir = Path(config.checkpoint_dir)
        if checkpoint_dir.exists():
            logger.info("Loading trained models...")
            system.load_models()
        
        # Create test dataset
        dataset = ImageDataset(
            args.data_dir,
            Path(args.labels) if args.labels else None
        )
        
        # Evaluate all models
        logger.info("Evaluating models on test dataset...")
        evaluation_results = system.evaluate_dataset(dataset)
        
        # Initialize analyzer
        analyzer = ModelPerformanceAnalyzer(args.output_dir)
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        comparison_report = analyzer.compare_models(evaluation_results)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        if 'individual_metrics' in evaluation_results:
            print("\n## Individual Model Performance:")
            for model_name, metrics in evaluation_results["individual_metrics"].items():
                print(f"\n{model_name}:")
                print(f"  - Accuracy: {metrics['accuracy']:.2%}")
                print(f"  - F1 Score: {metrics['f1_score']:.2%}")
        
        if 'ensemble_metrics' in evaluation_results:
            print("\n## Ensemble Performance:")
            ensemble_metrics = evaluation_results["ensemble_metrics"]
            print(f"  - Accuracy: {ensemble_metrics['accuracy']:.2%}")
            print(f"  - F1 Score: {ensemble_metrics['f1_score']:.2%}")
            
            if 'ensemble_improvement' in comparison_report:
                print(f"\n## Ensemble Improvement:")
                print(f"  - Absolute: +{comparison_report['ensemble_improvement']['absolute']:.3f}")
                print(f"  - Percentage: +{comparison_report['ensemble_improvement']['percentage']:.1f}%")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Creating visualizations...")
            analyzer.visualize_comparison(evaluation_results)
            print(f"\nVisualizations saved to {args.output_dir}")
        
        # Generate detailed report
        logger.info("Generating detailed performance report...")
        report_text = analyzer.generate_performance_report(evaluation_results)
        print(f"\nDetailed reports saved to {args.output_dir}")
    
    elif args.command == 'experiment':
        # Run full experiment (train + evaluate + compare)
        logger.info("Running full experiment...")
        
        # Import here to avoid circular imports
        from src.utils import ImageDataset
        
        # Train phase
        print("\n" + "="*60)
        print("PHASE 1: TRAINING")
        print("="*60)
        
        train_dataset = ImageDataset(
            args.train_dir,
            Path(args.labels) if args.labels else None
        )
        
        system.train(train_dataset, config.training)
        system.save_models()
        logger.info("Training completed")
        
        # Evaluate phase
        print("\n" + "="*60)
        print("PHASE 2: EVALUATION")
        print("="*60)
        
        test_dataset = ImageDataset(
            args.test_dir,
            Path(args.labels) if args.labels else None
        )
        
        evaluation_results = system.evaluate_dataset(test_dataset)
        
        # Compare phase
        print("\n" + "="*60)
        print("PHASE 3: MODEL COMPARISON")
        print("="*60)
        
        analyzer = ModelPerformanceAnalyzer("./results")
        comparison_report = analyzer.compare_models(evaluation_results)
        
        # Print final summary
        if 'ensemble_metrics' in evaluation_results:
            print(f"\nFinal Ensemble Performance: {evaluation_results['ensemble_metrics']['f1_score']:.2%} F1 Score")
        
        if args.visualize:
            analyzer.visualize_comparison(evaluation_results)
            print("\nVisualizations created in ./results/")
        
        # Generate final report
        report_text = analyzer.generate_performance_report(evaluation_results)
        print("\nExperiment completed! Results saved to ./results/")
    
    elif args.command == 'serve':
        # Start API server
        logger.info(f"Starting API server on {args.host}:{args.port}")
        
        # Import API module
        try:
            from src.api.server import create_app, run_server
            
            # Update config with command line args
            config.api_host = args.host
            config.api_port = args.port
            config.api_workers = args.workers
            
            # Create and run app
            app = create_app(config)
            run_server(app, host=args.host, port=args.port, workers=args.workers)
            
        except ImportError as e:
            logger.error(f"Failed to import API modules: {e}")
            logger.error("Make sure FastAPI is installed: pip install fastapi uvicorn")
            sys.exit(1)


if __name__ == "__main__":
    main()