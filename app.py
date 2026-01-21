import sys
from pathlib import Path
import numpy as np

from pipeline.base_handler import BaseHandler
from pipeline.handlers import (
    DataLoaderHandler,
    DataCleaningHandler,
    FeatureExtractionHandler,
    MissingDataHandler,
    OutlierRemovalHandler,
    CategoryGroupingHandler,
    EncodingHandler,
    NormalizationHandler,
    ArrayConversionHandler
)


def build_pipeline(csv_path: str) -> BaseHandler:
    """
    Build the data processing pipeline using Chain of Responsibility pattern

    Args:
        csv_path: Path to the input CSV file

    Returns:
        The first handler in the chain
    """
    # Create all handlers
    loader = DataLoaderHandler(csv_path)
    cleaner = DataCleaningHandler()
    feature_extractor = FeatureExtractionHandler()
    missing_handler = MissingDataHandler()
    outlier_remover = OutlierRemovalHandler(columns=['Age', 'Experience_Years'], iqr_multiplier=1.5)
    category_grouper = CategoryGroupingHandler(min_frequency=100)
    encoder = EncodingHandler()
    normalizer = NormalizationHandler()
    array_converter = ArrayConversionHandler()

    # Chain handlers together
    loader.set_next(cleaner) \
          .set_next(feature_extractor) \
          .set_next(missing_handler) \
          .set_next(outlier_remover) \
          .set_next(category_grouper) \
          .set_next(encoder) \
          .set_next(normalizer) \
          .set_next(array_converter)

    return loader


def save_arrays(X: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    """
    Save processed data as numpy arrays

    Args:
        X: Feature matrix
        y: Target vector
        output_dir: Directory to save the arrays
    """
    x_path = output_dir / 'x_data.npy'
    y_path = output_dir / 'y_data.npy'

    np.save(x_path, X)
    np.save(y_path, y)

    print(f"\n{'='*60}")
    print(f"Successfully saved processed data:")
    print(f"  X data: {x_path} (shape: {X.shape})")
    print(f"  y data: {y_path} (shape: {y.shape})")
    print(f"{'='*60}")


def main() -> None:
    """Main application entry point"""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Error: Invalid number of arguments", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} path/to/hh.csv", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    # Validate input file
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not csv_path.is_file():
        print(f"Error: Not a file: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if csv_path.suffix.lower() != '.csv':
        print(f"Warning: File does not have .csv extension: {csv_path}", file=sys.stderr)

    # Get output directory (same as input file directory)
    output_dir = csv_path.parent

    print(f"\n{'='*60}")
    print(f"HH.ru Data Processing Pipeline")
    print(f"{'='*60}")
    print(f"Input file: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    try:
        # Build and execute pipeline
        pipeline = build_pipeline(str(csv_path))
        X, y = pipeline.handle(None)

        # Save results
        save_arrays(X, y, output_dir)

    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
