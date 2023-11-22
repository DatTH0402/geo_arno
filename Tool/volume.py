import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
from fiona.drvsupport import supported_drivers
from simplekml import Kml, Color
supported_drivers['KML'] = 'rw'
supported_drivers['libkml'] = 'rw' # enable KML support which is disabled by default
supported_drivers['LIBKML'] = 'rw' # enable KML support which is disabled 
import numpy as np
import geopandas as gpd
from simplekml import Kml, Color
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import os
import time

start_time = time.time()

def hex_to_rgba(hex_code, alpha):
    r = int(hex_code[1:3], 16)
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)
    return (r, g, b, alpha)

def save_volume_data(dataframe, file_name):
    df = dataframe.copy()
    df['polygon_str'] = df['polygon_str'].str.replace('POLYGON ', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df['geometry'] = df['polygon_str'].apply(lambda x: Polygon([tuple(map(float, c.split())) for c in x.split(',')]))

    # Define the thresholds and colors for the custom colormap
    thresholds = [-1, 1, 2, 4, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, float('inf')]
    color_list = a = ['#fff7f7','#ffdfdf','#ffcfcf','#ffefef','#ffbfbf','#ffafaf','#ff9f9f','#ff8f8f','#ff7f7f','#ff6f6f','#ff5f5f','#ff4f4f','#ff3f3f','#ff2020','#ff0000','#ef0000','#df0000','#cf0000','#bf0000','#af0000','#9f0000','#8f0000','#7f0000','#6f0000','#5f0000','#4f0000','#3f0000','#2f0000','#1f0000','#0f0000']


    cmap = ListedColormap(color_list)
    df['color'] = pd.cut(df['Volume'], bins=thresholds, labels=color_list)
    # Convert the hexadecimal color codes to RGBA values
    rgba_colors = [hex_to_rgba(color,alpha=int(0.8*256)) for color in df['color']]

    # Create the GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')[['Volume','Samples','geometry']]

    # Create a KML object
    kml = Kml()
    # Iterate over the polygons in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # Add description to polygon
        attributes = row[['Volume','Samples']]  # Adjust the column names as per your data
        description = '<table>'
        table_html = """
            <style>
            table {
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
            }
            </style>
            <table >
                <tr>
                    <th>Metrics</th>
                    <th>Value</th>
                </tr>
        """
        for col, value in attributes.items():
            table_html += f'<tr><td>{col}</td><td>{value}</td></tr>'
        table_html += '</table>'
        
        ##
        polygon = kml.newpolygon(name=str(idx), outerboundaryis=row['geometry'].exterior.coords[:])
        polygon.description = table_html
        color = rgba_colors[idx]
        polygon.style.polystyle.color = Color.rgb(*color)
        polygon.style.linestyle.width = 1
        polygon.style.linestyle.color = Color.rgb(*color)

    # Save the KML file
    kml.savekmz(f"{file_name}.kmz")

def generate_bins(geom, bin_size=10):

    xmin, ymin, xmax, ymax = geom.bounds
    # Calculate the number of bins in each dimension
    xbins = int((xmax - xmin) / bin_size)
    ybins = int((ymax - ymin) / bin_size)
    # Create arrays of x and y coordinates for bins
    x_coords = np.arange(xmin, xmin + xbins * bin_size, bin_size)
    y_coords = np.arange(ymin, ymin + ybins * bin_size, bin_size)
    # Use NumPy's meshgrid to generate all combinations of x and y coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    # Flatten the arrays and create bin polygons using vectorized operations
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    bin_polygons = [
        Polygon([
            (x, y), (x + bin_size, y), (x + bin_size, y + bin_size), (x, y + bin_size)
        ]) for x, y in zip(x_flat, y_flat)
    ]
    return bin_polygons

def generate_volume_output(area_name, date_str, polygon_df, bin_size, df_raw, output_dir):
    print(f"***===RUNNING:{area_name}, {bin_size}*{bin_size} grid ==***")
    ### Read a Polygon and divide
    polygon_df = polygon_df.to_crs('EPSG:3857')
    bins_polygons = [generate_bins(geom, bin_size) for geom in polygon_df['geometry']]
    bins_polygons = [bin_poly for sublist in bins_polygons for bin_poly in sublist]
    # Create the GeoDataFrame for bins
    bins_df = gpd.GeoDataFrame({'geometry': bins_polygons}, crs=polygon_df.crs)
    bins_df_4326 = bins_df.to_crs("EPSG:4326")
    print('==Complete dividing bins')
    ### Load raw data file
    df_point = df_raw.copy()
    df_point_columns = [
    'Latitude',
    'Longitude',
    'UL Volume (kB)',
    'DL Volume (kB)',
    ]
    df_point = df_point[df_point_columns]

    ### Locate bins for each sample
    bins_df_4326.reset_index(inplace=True)
    polygonDF = bins_df_4326.copy()
    polygonDF = polygonDF[['geometry']]
    # Create Point Dataframe
    geometric_points = []
    for xy in zip(df_point['Longitude'], df_point['Latitude']):
        geometric_points.append(Point(xy))

    pointDF = gpd.GeoDataFrame(df_point,
                                    crs = {'init': 'epsg:4326'}, 
                                    geometry = geometric_points
                                    )
    pointDF = pointDF.to_crs(polygonDF.crs)
    # Join 2 DFs
    joinDF = gpd.sjoin(pointDF,polygonDF, how='inner', predicate='within')
    joinDF['polygon'] = joinDF['index_right'].map(polygonDF['geometry'])
    print('==Complete Locating point')

    ### Preprocessing
    df_segment = joinDF.copy()
    df_segment = df_segment[df_segment['polygon'].notna()]
    df_segment['polygon_str'] = df_segment['polygon'].astype(str)

    ## Volume Data (UL+DL)
    df5 = df_segment[['polygon_str','UL Volume (kB)','DL Volume (kB)']].copy()
    df5['UL Volume (kB)'] = df5['UL Volume (kB)'].str.replace(',', '').astype(float)
    df5['DL Volume (kB)'] = df5['DL Volume (kB)'].str.replace(',', '').astype(float)
    df5['Volume'] = (df5['UL Volume (kB)'] + df5['DL Volume (kB)'])/1000
    df5 = df5[['polygon_str', 'Volume']]
    df5_sum = df5.groupby(['polygon_str']).sum()
    df5_count = df5.groupby(['polygon_str']).count()
    df5 = df5_sum
    df5['Samples'] = df5_count['Volume']
    df5.reset_index(inplace=True)
    output_volume_kml = os.path.join(output_dir, area_name+'_Volume_'+str(bin_size)+'_'+date_str)
    save_volume_data(df5, output_volume_kml)
    print('==Complete Saving Volume Data')
    ###
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"===Process completed: {area_name} , {bin_size}*{bin_size} ===")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"*****===== PROCESS COMPLETED: {area_name}, {bin_size}*{bin_size} grid =====*****")