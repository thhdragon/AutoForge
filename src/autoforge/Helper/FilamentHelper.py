import sys
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
import json


def load_materials(args):
    """
    Load material data from a CSV file.

    Args:
        csv_filename (str): Path to the hueforge CSV file containing material data.

    Returns:
        tuple: A tuple containing:
            - material_colors (jnp.ndarray): Array of material colors in float64.
            - material_TDs (jnp.ndarray): Array of material transmission/opacity parameters in float64.
            - material_names (list): List of material names.
            - colors_list (list): List of color hex strings.

    Raises:
        ValueError: If Transmissivity values are invalid (≤ 0).
    """
    df = load_materials_pandas(args)
    material_names = [
        str(brand) + " - " + str(name)
        for brand, name in zip(df["Brand"].tolist(), df["Name"].tolist())
    ]
    material_TDs = (df["Transmissivity"].astype(float)).to_numpy()
    colors_list = df["Color"].tolist()
    # Use float64 for material colors.
    material_colors = np.array(
        [hex_to_rgb(color) for color in colors_list], dtype=np.float64
    )
    material_TDs = np.array(material_TDs, dtype=np.float64)

    # Validate Transmissivity values: must be positive to avoid division by zero
    # in opacity calculations (thick_ratio = thickness / TD)
    invalid_mask = material_TDs <= 0
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = material_TDs[invalid_mask]
        invalid_materials = [material_names[i] for i in invalid_indices]
        raise ValueError(
            f"Invalid Transmissivity values in CSV (must be > 0):\n"
            f"  Materials: {invalid_materials}\n"
            f"  Values: {invalid_values}\n"
            f"Please check your CSV file and ensure all Transmissivity values are positive."
        )

    return material_colors, material_TDs, material_names, colors_list


def load_materials_pandas(args):
    csv_filename = args.csv_file
    json_filename = args.json_file

    if csv_filename != "":
        try:
            df = pd.read_csv(csv_filename)
        except Exception as e:
            traceback.print_exc()
            print("Error reading filament CSV file:", e)
            sys.exit(1)
        # rename all columns that start with a whitespace
        df.columns = [col.strip() for col in df.columns]
        # if TD in columns rename to Transmissivity
        if "TD" in df.columns:
            df.rename(columns={"TD": "Transmissivity"}, inplace=True)
    else:
        # read json
        with open(json_filename, "r") as f:
            data = json.load(f)
        if "Filaments" in data.keys():
            data = data["Filaments"]
        else:
            print(
                "Warning: No Filaments key found in JSON data. We can't use this json data."
            )
            sys.exit(1)
        # list to dataframe
        df = pd.DataFrame(data)
    return df


def load_materials_data(args):
    """
    Load the full material data from the CSV file.

    Args:
        csv_filename (str): Path to the CSV file containing material data.

    Returns:
        list: A list of dictionaries (one per material) with keys such as
              "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = load_materials_pandas(args)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Args:
        hex_str (str): The hex color string (e.g., '#RRGGBB' or '#RGB').

    Returns:
        list: A list of three floats representing the RGB values normalized to [0, 1].

    Raises:
        ValueError: If hex_str is not a valid hex color format.
    """
    hex_str = hex_str.lstrip("#")

    # Support 3-char hex: #ABC -> #AABBCC
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])

    # Validate length
    if len(hex_str) != 6:
        raise ValueError(
            f"Invalid hex color length: #{hex_str} (expected 6 or 3 characters)"
        )

    # Validate and convert hex digits with better error messages
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return [r / 255.0, g / 255.0, b / 255.0]
    except ValueError:
        raise ValueError(
            f"Invalid hex digits in color: #{hex_str} (must contain only 0-9, A-F)"
        )


def extract_colors_from_swatches(swatch_data):
    # we keep only data with transmission distance
    swatch_data = [swatch for swatch in swatch_data if swatch["td"]]

    # For now we load it and convert it in the same way as the hueforge csv files
    out = {}
    for swatch in swatch_data:
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        color = swatch["hex_color"]
        td = swatch["td"]
        out[(brand, name)] = (color, td)

    # convert to the same format as the hueforge csv files
    material_names = [str(brand) + " - " + str(name) for (brand, name) in out.keys()]
    material_colors = np.array(
        [hex_to_rgb("#" + color) for color, _ in out.values()], dtype=np.float64
    )
    material_TDs = np.array([td for _, td in out.values()], dtype=np.float64)
    colors_list = [color for color, _ in out.values()]

    return material_colors, material_TDs, material_names, colors_list


def swatch_data_to_table(swatch_data):
    """
    Converts swatch JSON data into a table (list of dicts) with columns:
    "Brand", "Name", "Transmission Distance", "Hex Color".
    """
    table = []
    for swatch in swatch_data:
        if not swatch["td"]:
            continue
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        hex_color = swatch["hex_color"]
        td = swatch["td"]
        table.append(
            {
                "Brand": brand,
                "Name": name,
                "TD": td,
                "HexColor": f"#{hex_color}",
                "Uuid": str(uuid.uuid4()),
            }
        )
    return table


def count_distinct_colors(dg: torch.Tensor) -> int:
    """
    Count how many distinct color/material IDs appear in dg.
    """
    unique_mats = torch.unique(dg)
    return len(unique_mats)


def count_swaps(dg: torch.Tensor) -> int:
    """
    Count how many color changes (swaps) occur between adjacent layers.
    """
    # A 'swap' is whenever dg[i] != dg[i+1].
    return int((dg[:-1] != dg[1:]).sum().item())
