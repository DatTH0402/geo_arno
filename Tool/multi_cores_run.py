import os
import sys
import warnings
from pilot import generate_pilot_output
from volume import generate_volume_output
import geopandas as gpd
import pandas as pd
import multiprocessing


def check_file_exist(file_path):
    if not os.path.exists(file_path):
        warnings.warn(f"File '{file_path}' Không tồn tại.", UserWarning)

def run_pilot(area_name, date_str, polygon_df, bin_size, df_raw):
    
    output_dir = f"../Result/{area_name}/Bin_{bin_size}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    generate_pilot_output(area_name=area_name,
        date_str=date_str,
        polygon_df= polygon_df,
        bin_size=bin_size,
        df_raw=df_raw,
        output_dir=output_dir)
    
def run_volume(area_name, date_str, polygon_df, bin_size, df_raw):
    output_dir = f"../Result/{area_name}/Bin_{bin_size}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    generate_volume_output(area_name=area_name,
        date_str=date_str,
        polygon_df= polygon_df,
        bin_size=bin_size,
        df_raw=df_raw,
        output_dir=output_dir)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Cách chạy tool: python run.py tên_khu_vực ngày ===> ví dụ: python run.py DN_KCN_AN_DON 050723")
        sys.exit(1)

    area_name = sys.argv[1]
    date_str = sys.argv[2]
    run_type = sys.argv[3]

    input_polygon_file = f"../Polygon/{area_name}.kml"
    polygon_df = gpd.read_file(input_polygon_file)
    print('--Reading input area polygon and divide bins: ', input_polygon_file)
    check_file_exist(input_polygon_file)

    if(run_type=='pilot'):
        folder_path = f"../Raw/{area_name}/Pilot_{date_str}"
    if(run_type=='volume'):
        folder_path = f"../Raw/{area_name}/Data_{date_str}"
    print(f"Reading from multiple input raw files in {folder_path}")
    df_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df_list.append(df)
    df_raw = pd.concat(df_list, ignore_index=True)

    data = [(area_name, date_str, polygon_df, 10, df_raw),
            (area_name, date_str, polygon_df, 20, df_raw),
            (area_name, date_str, polygon_df, 50, df_raw)]
    num_processes = 3
    pool = multiprocessing.Pool(processes=num_processes)
    if(run_type=='pilot'):
        results = pool.starmap(run_pilot, data)
    if(run_type=='volume'):
        results = pool.starmap(run_volume, data)
    pool.close()