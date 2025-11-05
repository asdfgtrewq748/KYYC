import os
import re
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
TEST_BH_DIR = os.path.join(DATA_DIR, '测试钻孔')
COORD_FILE = os.path.join(DATA_DIR, 'zuobiao.csv')
OUT_FILE = os.path.join(DATA_DIR, 'geology_features_extracted.csv')

def read_borehole_csv(path):
    # read with pandas, try to handle encoding and extra columns
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(path, encoding='gbk')
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # common column names in provided data
    # 序号(从下到上),名称,厚度/m,弹性模量/Gpa,容重/kN*m-3,抗拉强度/MPa
    col_map = {}
    for c in df.columns:
        low = c.lower()
        if '序' in low and '下' in low:
            col_map[c] = 'index_from_bottom'
        elif '名' in low:
            col_map[c] = 'name'
        elif '厚' in low:
            col_map[c] = 'thickness'
        elif '弹性' in low or '弹' in low:
            col_map[c] = 'elastic_modulus'
        elif '容重' in low or '密度' in low:
            col_map[c] = 'density'
        elif '抗拉' in low or '强度' in low:
            col_map[c] = 'tensile_strength'
    df = df.rename(columns=col_map)
    # keep only relevant columns
    for col in ['index_from_bottom','name','thickness','elastic_modulus','density','tensile_strength']:
        if col not in df.columns:
            df[col] = np.nan
    # convert numeric columns
    df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce').fillna(0.0)
    df['elastic_modulus'] = pd.to_numeric(df['elastic_modulus'], errors='coerce')
    df['density'] = pd.to_numeric(df['density'], errors='coerce')
    df['tensile_strength'] = pd.to_numeric(df['tensile_strength'], errors='coerce')
    # ensure index ordering: index_from_bottom numeric
    try:
        df['index_from_bottom'] = pd.to_numeric(df['index_from_bottom'], errors='coerce')
    except Exception:
        df['index_from_bottom'] = np.arange(len(df))
    # Sort by index_from_bottom ascending => bottom(0) -> top(N)
    df = df.sort_values('index_from_bottom', ascending=True).reset_index(drop=True)
    return df


def extract_features_from_bh(df):
    total_thickness = df['thickness'].sum()
    # weighted averages by thickness when available
    def wavg(values, weights):
        mask = (~values.isna()) & (weights > 0)
        if mask.sum() == 0:
            return np.nan
        return (values[mask] * weights[mask]).sum() / weights[mask].sum()

    avg_elastic = wavg(df['elastic_modulus'], df['thickness'])
    avg_density = wavg(df['density'], df['thickness'])
    max_tensile = df['tensile_strength'].max()
    # lithology categories
    names = df['name'].astype(str).fillna('')
    is_coal = names.str.contains('煤') | names.str.contains('coal', case=False)
    coal_thickness = df.loc[is_coal, 'thickness'].sum()
    coal_count = int(is_coal.sum())
    # proportion of sandstones and mudstones
    is_sand = names.str.contains('砂') | names.str.contains('砂岩') | names.str.contains('sand', case=False)
    is_mud = names.str.contains('泥') | names.str.contains('泥岩') | names.str.contains('mud', case=False)
    sand_thick = df.loc[is_sand, 'thickness'].sum()
    mud_thick = df.loc[is_mud, 'thickness'].sum()
    prop_sand = sand_thick / total_thickness if total_thickness > 0 else np.nan
    prop_mud = mud_thick / total_thickness if total_thickness > 0 else np.nan
    # depth to first coal from top: we need cumulative thickness from top
    # since df sorted bottom->top, cumulative from top = total_thickness - cumsum(thickness) + thickness_of_layer
    # find topmost coal layer index (highest index_from_bottom)
    if coal_count > 0:
        # find first coal from top => the coal with largest index_from_bottom
        coal_rows = df[is_coal]
        topmost_coal_idx = coal_rows['index_from_bottom'].max()
        # compute depth from top to top of that coal seam: sum of thickness of layers above it
        # layers above = rows with index_from_bottom > topmost_coal_idx
        layers_above = df[df['index_from_bottom'] > topmost_coal_idx]
        depth_to_top_coal = layers_above['thickness'].sum()
    else:
        depth_to_top_coal = np.nan
    return {
        'total_thickness_m': total_thickness,
        'coal_thickness_m': coal_thickness,
        'coal_seam_count': coal_count,
        'depth_to_top_coal_m': depth_to_top_coal,
        'avg_elastic_modulus_GPa': avg_elastic,
        'avg_density_kN_m3': avg_density,
        'max_tensile_MPa': max_tensile,
        'prop_sandstone': prop_sand,
        'prop_mudstone': prop_mud
    }


def main():
    # read coordinates
    coords = pd.read_csv(COORD_FILE, encoding='utf-8')
    coords.columns = [c.strip() for c in coords.columns]
    # standard column names: 钻孔名,坐标x,坐标y
    coords = coords.rename(columns={coords.columns[0]: 'borehole', coords.columns[1]: 'x', coords.columns[2]: 'y'})
    coords['borehole'] = coords['borehole'].astype(str).str.strip()

    results = []
    # iterate BK-*.csv files
    pattern = os.path.join(TEST_BH_DIR, 'BK-*.csv')
    files = glob.glob(pattern)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = read_borehole_csv(f)
            feats = extract_features_from_bh(df)
            feats['borehole'] = name
            results.append(feats)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
    feats_df = pd.DataFrame(results)
    merged = pd.merge(coords, feats_df, left_on='钻孔名' if '钻孔名' in coords.columns else 'borehole', right_on='borehole', how='left')
    # make sure x,y columns exist
    if 'x' not in merged.columns and '坐标x' in merged.columns:
        merged = merged.rename(columns={'坐标x': 'x', '坐标y': 'y'})
    # save
    merged.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print('Saved features to', OUT_FILE)

if __name__ == '__main__':
    main()
