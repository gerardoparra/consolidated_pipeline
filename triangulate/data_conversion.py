import pandas as pd
import os
from pathlib import Path
import glob
import numpy as np

def convert_superanimal_to_anipose(input_csv_path, output_csv_path):
    """
    Convert SuperAnimal CSV format to Anipose-compatible format.
    
    SuperAnimal format: 4 header rows (model, individual, bodypart, coord)
    Anipose format: 3 header rows (scorer, bodyparts, coords)
    """
    
    # Read the CSV with all header rows
    # SuperAnimal has 4 header rows
    df = pd.read_csv(input_csv_path, header=[0, 1, 2, 3])
    
    # Get the multi-index columns
    columns = df.columns
    
    # Create new multi-index for Anipose format
    new_columns = []
    
    for col in columns:
        # SuperAnimal structure: (model, individual, bodypart, coord)
        model = col[0]      # Row 1: model name
        individual = col[1]  # Row 2: animal ID
        bodypart = col[2]    # Row 3: body part name
        coord = col[3]       # Row 4: x/y/likelihood
        
        # Anipose structure: (scorer, bodyparts, coords)
        # Use model name as scorer, bodypart from row 3
        new_col = (model, bodypart, coord)
        new_columns.append(new_col)
    
    # Assign new columns to dataframe
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    
    # CRITICAL: Set the column index names as Anipose expects
    df.columns.names = ['scorer', 'bodyparts', 'coords']
    
    # Save to new CSV (overwrite if exists)
    df.to_csv(output_csv_path, index=False)
    print(f"Converted CSV: {input_csv_path} -> {output_csv_path}")
    
    return df

def convert_superanimal_h5_to_anipose(input_h5_path, output_h5_path):
    """
    Convert SuperAnimal H5 file to Anipose-compatible H5 format.
    Preserves H5 format but restructures the multi-index columns.
    """
    try:
        # Read H5 file
        df = pd.read_hdf(input_h5_path)
        
        # Check if it has the expected 4-level columns
        if df.columns.nlevels == 4:
            # Convert to Anipose format (3 levels)
            new_columns = []
            for col in df.columns:
                # Assuming structure: (model, individual, bodypart, coord)
                model = col[0]
                bodypart = col[2]  # Skip the individual level
                coord = col[3]
                new_col = (model, bodypart, coord)
                new_columns.append(new_col)
            
            df.columns = pd.MultiIndex.from_tuples(new_columns)
            
        elif df.columns.nlevels == 3:
            # Check if it already has the right structure but might need index names
            print(f"File {input_h5_path} already has 3 header levels")
            
        else:
            print(f"Warning: {input_h5_path} has {df.columns.nlevels} header levels, attempting conversion")
            if df.columns.nlevels > 3:
                new_columns = []
                for col in df.columns:
                    # Take last 3 levels if there are more
                    model = col[-3] if len(col) >= 3 else col[0]
                    bodypart = col[-2] if len(col) >= 2 else col[1]
                    coord = col[-1]
                    new_col = (model, bodypart, coord)
                    new_columns.append(new_col)
                df.columns = pd.MultiIndex.from_tuples(new_columns)
        
        # CRITICAL: Set the column index names as Anipose expects
        df.columns.names = ['scorer', 'bodyparts', 'coords']
        
        # Save as H5 (overwrite if exists)
        df.to_hdf(output_h5_path, key='df', mode='w')
        print(f"Converted H5: {input_h5_path} -> {output_h5_path}")
        return df
        
    except Exception as e:
        print(f"Error converting H5 {input_h5_path}: {e}")
        return None

def batch_convert_superanimal_to_anipose(input_folder, output_folder, 
                                        input_pattern="*.csv", 
                                        also_convert_h5=True,
                                        overwrite=True):
    """
    Batch convert all SuperAnimal files in a folder to Anipose format.
    
    Parameters:
    -----------
    input_folder : str or Path
        Folder containing SuperAnimal output files
    output_folder : str or Path
        Folder where converted files will be saved
    input_pattern : str
        Pattern to match files (default: "*.csv")
    also_convert_h5 : bool
        Whether to also convert H5 files (default: True)
    overwrite : bool
        Whether to overwrite existing output files (default: True)
    """
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    converted_csv_count = 0
    converted_h5_count = 0
    
    # Convert CSV files
    csv_files = glob.glob(str(Path(input_folder) / input_pattern))
    
    for csv_file in csv_files:
        try:
            # Generate output filename (same name, same extension)
            input_path = Path(csv_file)
            output_file = output_path / input_path.name
            
            # Skip if output exists and overwrite is False
            if not overwrite and output_file.exists():
                print(f"Skipping existing file: {output_file}")
                continue
            
            # Convert the file
            convert_superanimal_to_anipose(csv_file, output_file)
            converted_csv_count += 1
            
        except Exception as e:
            print(f"Error converting {csv_file}: {e}")
    
    # Convert H5 files if requested
    if also_convert_h5:
        h5_files = glob.glob(str(Path(input_folder) / "*.h5"))
        
        for h5_file in h5_files:
            try:
                # Generate output filename (same name, same extension)
                input_path = Path(h5_file)
                output_file = output_path / input_path.name
                
                # Skip if output exists and overwrite is False
                if not overwrite and output_file.exists():
                    print(f"Skipping existing file: {output_file}")
                    continue
                
                convert_superanimal_h5_to_anipose(h5_file, output_file)
                converted_h5_count += 1
                
            except Exception as e:
                print(f"Error converting {h5_file}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Converted {converted_csv_count} CSV files")
    if also_convert_h5:
        print(f"Converted {converted_h5_count} H5 files")
    print(f"Output saved to: {output_folder}")
    print(f"{'='*50}")

def verify_conversion(input_path, output_path, file_type='csv'):
    """
    Verify that the conversion worked correctly for Anipose.
    """
    print(f"\n{'='*50}")
    print(f"Verifying conversion for: {Path(input_path).name}")
    print(f"{'='*50}")
    
    if file_type == 'csv':
        df_conv = pd.read_csv(output_path, header=[0, 1, 2])
    else:  # h5
        df_conv = pd.read_hdf(output_path)
    
    print(f"Converted columns levels: {df_conv.columns.nlevels}")
    print(f"Column index names: {df_conv.columns.names}")
    
    # Check if column names are correct for Anipose
    expected_names = ['scorer', 'bodyparts', 'coords']
    has_correct_names = all(name in df_conv.columns.names for name in expected_names)
    
    print(f"Has correct index names ('scorer', 'bodyparts', 'coords'): {has_correct_names}")
    
    # Check if 'neck_base' exists at the expected level
    if has_correct_names and df_conv.columns.nlevels >= 2:
        bodyparts = df_conv.columns.get_level_values('bodyparts').unique()
        neck_base_present = 'neck_base' in bodyparts
        
        print(f"'neck_base' present in converted file: {neck_base_present}")
        
        if neck_base_present:
            # Show first few bodyparts
            print(f"\nFirst 10 bodyparts in converted file:")
            for i, bp in enumerate(bodyparts[:10]):
                print(f"  {i+1}. {bp}")
        
        # Also check for likelihood column
        coords = df_conv.columns.get_level_values('coords').unique()
        print(f"\nCoordinate types present: {list(coords)}")
        
    else:
        print("WARNING: Column index names are not correctly set!")
        print(f"Current names: {df_conv.columns.names}")
        print(f"Expected names: {expected_names}")
    
    return has_correct_names

def batch_verify_conversion(input_folder, output_folder, also_convert_h5=True):
    """
    Verify conversion for all files in the output folder.
    """
    output_path = Path(output_folder)
    all_valid = True
    
    # Verify CSV files
    csv_files = list(output_path.glob("*.csv"))
    for csv_file in csv_files:
        input_file = Path(input_folder) / csv_file.name
        if input_file.exists():
            valid = verify_conversion(input_file, csv_file, 'csv')
            all_valid = all_valid and valid
    
    # Verify H5 files
    if also_convert_h5:
        h5_files = list(output_path.glob("*.h5"))
        for h5_file in h5_files:
            input_file = Path(input_folder) / h5_file.name
            if input_file.exists():
                valid = verify_conversion(input_file, h5_file, 'h5')
                all_valid = all_valid and valid
    
    if all_valid:
        print("\n✓ All files are correctly formatted for Anipose!")
    else:
        print("\n✗ Some files have formatting issues. Please check the output above.")
    
    return all_valid

def fix_existing_converted_files(folder_path, file_type='both'):
    """
    Fix already converted files by adding the correct column index names.
    Useful if you already ran the conversion without the index names.
    
    Parameters:
    -----------
    folder_path : str or Path
        Folder containing already converted files
    file_type : str
        'csv', 'h5', or 'both'
    """
    folder = Path(folder_path)
    fixed_count = 0
    
    if file_type in ['csv', 'both']:
        csv_files = list(folder.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, header=[0, 1, 2])
                if df.columns.names != ['scorer', 'bodyparts', 'coords']:
                    df.columns.names = ['scorer', 'bodyparts', 'coords']
                    df.to_csv(csv_file, index=False)
                    print(f"Fixed CSV: {csv_file.name}")
                    fixed_count += 1
                else:
                    print(f"CSV already correct: {csv_file.name}")
            except Exception as e:
                print(f"Error fixing {csv_file}: {e}")
    
    if file_type in ['h5', 'both']:
        h5_files = list(folder.glob("*.h5"))
        for h5_file in h5_files:
            try:
                df = pd.read_hdf(h5_file)
                if df.columns.names != ['scorer', 'bodyparts', 'coords']:
                    df.columns.names = ['scorer', 'bodyparts', 'coords']
                    df.to_hdf(h5_file, key='df', mode='w')
                    print(f"Fixed H5: {h5_file.name}")
                    fixed_count += 1
                else:
                    print(f"H5 already correct: {h5_file.name}")
            except Exception as e:
                print(f"Error fixing {h5_file}: {e}")
    
    print(f"\nFixed {fixed_count} files in {folder}")
    return fixed_count