import numpy as np
import pandas as pd


def make_features(data: pd.DataFrame, variant: str = 'physics') -> pd.DataFrame:
    x = data.drop(columns=['id', 'target'], errors='ignore').copy()

    for column in ['material', 'printer_model', 'bed_surface', 'filament_color']:
        x[column] = x[column].fillna('Missing').astype(str)

    references = {
        'nozzle_temp_c': {'PLA': 210, 'PETG': 240, 'ABS': 250, 'TPU': 220},
        'bed_temp_c': {'PLA': 60, 'PETG': 75, 'ABS': 100, 'TPU': 50},
        'cooling_percent': {'PLA': 90, 'PETG': 40, 'ABS': 10, 'TPU': 50},
    }

    for column, values in references.items():
        relative = x[column] - x['material'].map(values)
        x[column + '_relative'] = relative
        x[column + '_deviation'] = relative.abs()

    x['layer_nozzle_ratio'] = x['layer_height_mm'] / x['nozzle_diameter_mm']
    x['mass_contact_ratio'] = x['model_mass_g'] / x['contact_area_cm2']
    x['flow'] = x['print_speed_mm_s'] * x['layer_height_mm'] * x['nozzle_diameter_mm']

    for offset in [-10, -5, 5, 10]:
        x[f'nozzle_abs_{offset}'] = (x['nozzle_temp_c_relative'] - offset).abs()
        x[f'bed_abs_{offset}'] = (x['bed_temp_c_relative'] - offset).abs()

    x['speed_material'] = x['print_speed_mm_s'] / x['material'].map({'PLA': 65, 'PETG': 50, 'ABS': 60, 'TPU': 25})
    x['adhesion_risk'] = x['model_mass_g'] / (x['contact_area_cm2'] + 1)
    x['overhang_cooling'] = x['overhang_angle_deg'] * (100 - x['cooling_percent']) / 100
    x['complex_speed'] = x['geometry_complexity'] * x['print_speed_mm_s']
    x['thin_contact'] = x['contact_area_cm2'] / x['model_mass_g'] ** (2 / 3)
    x['layer_complexity'] = x['layer_nozzle_ratio'] * x['geometry_complexity']
    x['time_moisture'] = np.log1p(x['estimated_time_min']) * x['filament_moisture_pct']

    if variant == 'extra':
        x['relative_flow'] = x['flow'] / x['material'].map({'PLA': 12, 'PETG': 10, 'ABS': 11, 'TPU': 4})
        x['complex_overhang'] = x['geometry_complexity'] * x['overhang_angle_deg']
        x['moisture_material'] = x['filament_moisture_pct'] * x['material'].map({'PLA': 1, 'PETG': 1.5, 'ABS': 1, 'TPU': 2})
        x['moisture_duration'] = x['filament_moisture_pct'] * x['estimated_time_min']
        x['relative_flow2'] = x['flow'] / x['material'].map({'PLA': 10, 'PETG': 8, 'ABS': 10, 'TPU': 3})
        x['mass_area'] = x['model_mass_g'] / (x['contact_area_cm2'] + 5)
        x['overhang_layer'] = x['overhang_angle_deg'] * x['layer_nozzle_ratio']

    return x.replace([np.inf, -np.inf], np.nan)
