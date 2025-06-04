#!/usr/bin/env python3
"""
Script to download astronomical data from CANFAR.
Usage: python dwarforge_download.py --tile TILE_ID
Example: python dwarforge_download.py --tile 319_280
"""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from vos import Client


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download data from CANFAR dwarforge project.')
    parser.add_argument('--tile', required=True, help='Tile ID, e.g., 319_280')
    parser.add_argument(
        '--workers', type=int, default=7, help='Number of parallel download workers'
    )
    return parser.parse_args()


def create_local_directory(directory):
    """Create local directory if it doesn't exist."""
    if not os.path.exists(directory):
        print(f'Creating directory: {directory}')
        os.makedirs(directory)
    else:
        print(f'Directory already exists: {directory}')


def check_file_exists_locally(local_path):
    """Check if a file exists locally."""
    return os.path.exists(local_path)


def download_file(source_path, dest_path):
    """Download a file using vcp."""
    print(f'Downloading: {source_path} -> {dest_path}')
    try:
        result = subprocess.run(
            ['vcp', source_path, dest_path], check=True, capture_output=True, text=True
        )
        return (True, source_path, dest_path)
    except subprocess.CalledProcessError as e:
        print(f'Error downloading {source_path}: {e}')
        if hasattr(e, 'stderr'):
            print(f'Standard error: {e.stderr}')
        return (False, source_path, dest_path)


def download_worker(download_item):
    """Worker function for ThreadPoolExecutor."""
    remote_path, local_path = download_item
    return download_file(remote_path, local_path)


def main():
    # Parse command line arguments
    args = parse_arguments()
    tile_id = args.tile
    num_workers = args.workers

    # Extract tile numbers for proper formatting
    tile_parts = tile_id.split('_')
    if len(tile_parts) != 2:
        print(f'Error: Tile ID should be in format XXX_YYY (e.g., 319_280), got {tile_id}')
        return

    tile_num1, tile_num2 = tile_parts

    # Initialize VOSpace client
    client = Client()

    # Define base directory and bands
    base_dir = 'arc:projects/unions/ssl/data/raw/tiles/dwarforge'
    tile_dir = f'{base_dir}/{tile_id}'

    bands = {
        'whigs-g': {
            'patterns': [
                f'calexp-CFIS_{tile_num1}_{tile_num2}_rebin*.fits',
                f'calexp-CFIS_{tile_num1}_{tile_num2}_rebin*params.parquet',
            ]
        },
        'cfis_lsb-r': {
            'patterns': [
                f'CFIS_LSB.{tile_num1}.{tile_num2}.r_rebin*.fits',
                f'CFIS_LSB.{tile_num1}.{tile_num2}.r*params.parquet',
            ]
        },
        'ps-i': {
            'patterns': [
                f'PS-DR3.{tile_num1}.{tile_num2}.i_rebin*.fits',
                f'PS-DR3.{tile_num1}.{tile_num2}.i*params.parquet',
            ]
        },
        'gri': {'patterns': [f'{tile_num1}_{tile_num2}_matched_detections*.parquet']},
    }

    # Create local tile directory
    local_tile_dir = f'test_fields/{tile_id}'
    create_local_directory(local_tile_dir)

    # Initialize counters for summary
    files_found = 0
    files_already_exist = 0
    files_downloaded = 0
    files_failed = 0

    # Find and download files
    files_to_download = []

    # Discover files to download
    for band_name, band_info in bands.items():
        # Remote band directory
        remote_band_dir = f'{tile_dir}/{band_name}'
        print(f'\nChecking band: {band_name}')

        for pattern in band_info['patterns']:
            try:
                print(f'  Searching with pattern: {pattern}')
                # Find files matching the pattern
                matching_files = client.glob1(remote_band_dir, pattern)

                if not matching_files:
                    print(f'  No files found matching pattern: {pattern}')
                    continue

                for file_name in matching_files:
                    files_found += 1
                    remote_path = f'{remote_band_dir}/{file_name}'
                    local_path = f'{local_tile_dir}/{file_name}'

                    # Check if file already exists locally
                    if check_file_exists_locally(local_path):
                        print(f'  File already exists locally: {file_name}')
                        files_already_exist += 1
                    else:
                        print(f'  Found file to download: {file_name}')
                        # Add to download list
                        files_to_download.append((remote_path, local_path))

            except Exception as e:
                print(f'  Error finding files in {remote_band_dir} with pattern {pattern}: {e}')

    # Download the missing files using parallel processing
    if files_to_download:
        print(
            f'\nDownloading {len(files_to_download)} files using {num_workers} parallel workers...'
        )

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all download tasks
            future_to_download = {
                executor.submit(download_worker, item): item for item in files_to_download
            }

            # Process completed downloads
            for future in as_completed(future_to_download):
                success, remote_path, local_path = future.result()
                if success:
                    files_downloaded += 1
                    print(f'Successfully downloaded: {os.path.basename(local_path)}')
                else:
                    files_failed += 1
                    print(f'Failed to download: {os.path.basename(local_path)}')
    else:
        print('\nNo files to download. All files already exist locally.')

    # Print summary
    print('\nSummary:')
    print(f'Total files found: {files_found}')
    print(f'Files already existing locally: {files_already_exist}')
    print(f'Files downloaded successfully: {files_downloaded}')
    print(f'Files failed to download: {files_failed}')


if __name__ == '__main__':
    main()
