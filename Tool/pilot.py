import math
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
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

start_time = time.time()

def hex_to_rgba(hex_code, alpha):
    r = int(hex_code[1:3], 16)
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)
    return (r, g, b, alpha)

def save_output_to_kml(dataframe, metric, theme, file_name):

    df = dataframe.copy()
    df['polygon_str'] = df['polygon_str'].str.replace('POLYGON ', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df['geometry'] = df['polygon_str'].apply(lambda x: Polygon([tuple(map(float, c.split())) for c in x.split(',')]))

    # Define the thresholds and colors for the custom colormap
    if theme == 'rsrp':
        thresholds = [-float('inf'), -118, -108, -98, -82, float('inf')]
        color_list = ['#000000', '#ff0000', '#ffff00', '#00ff00', '#1e90ff']
    if theme == 'pilot_pollution':
        thresholds = [-float('inf'), 0.33, 1.33, 2.33, 3.33, float('inf')]
        color_list = ['#1e90ff', '#00ff00', '#ffff00', '#ff0000', '#000000']

    
    cmap = ListedColormap(color_list)
    df['color'] = pd.cut(df[metric], bins=thresholds, labels=color_list)
    # Convert the hexadecimal color codes to RGBA values
    rgba_colors = [hex_to_rgba(color, alpha=int(0.8*256)) for color in df['color']]

    # Create the GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')[['bin_id',metric, 'No.Samples','geometry']]

    # Create a KML object
    kml = Kml()
    # Iterate over the polygons in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # Add description to polygon
        attributes = row[['bin_id',metric,'No.Samples']]  # Adjust the column names as per your data
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
        polygon.visibility = 0

    # Save the KML file
    kml.savekmz(file_name+'.kmz')

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
    gdf = gpd.GeoDataFrame(df, geometry='geometry')[['Volume','No.Samples','geometry']]

    # Create a KML object
    kml = Kml()
    # Iterate over the polygons in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # Add description to polygon
        attributes = row[['Volume','No.Samples']]  # Adjust the column names as per your data
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
        polygon.visibility = 0

    # Save the KML file
    kml.savekmz(f"{file_name}.kmz")

def generate_bins(geom, bin_size=10):

    x_ref, y_ref = 193602, 2602072
    xmin, ymin, xmax, ymax = geom.bounds
    # To shift the polygon to math the whole grid
    xmin = math.floor((xmin-x_ref)/bin_size)*bin_size+x_ref
    xmax = math.ceil((xmax-x_ref)/bin_size)*bin_size+x_ref
    ymin = math.floor((ymin-y_ref)/bin_size)*bin_size+y_ref
    ymax = math.ceil((ymax-y_ref)/bin_size)*bin_size+y_ref
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

def filter_cell_based_on_distance(df_segment, gdf_cell_locations):

    ## Count the occurrences of each cell in nth_best_cell columns
    best_cell_label_columns = [
        'Serving Cell Label',
        'Best Cell Label',
        'Second Best Cell Label',
        'Third Best Cell Label',
        'Fourth Best Cell Label',
        'Fifth Best Cell Label',
        'Sixth Best Cell Label',
    ]
    df_cell_counts = pd.DataFrame({'Cell': []})
    for column in best_cell_label_columns:
        counts = df_segment[column].value_counts().reset_index()
        counts.columns = ['Cell', column]
        df_cell_counts = pd.merge(df_cell_counts, counts, how='outer', on='Cell')
    df_cell_counts = df_cell_counts.fillna(0)
    df_cell_counts['Occurrences'] = df_cell_counts[best_cell_label_columns[1:]].sum(axis=1)
    df_cell_counts_return = df_cell_counts.copy()
    df_cell_counts = df_cell_counts[df_cell_counts['Occurrences'] <= 50]

    cells_to_check = set(df_cell_counts['Cell'])
    mask = df_segment[best_cell_label_columns[1:]].isin(cells_to_check).any(axis=1)
    df_cell_filtered = df_segment[mask][best_cell_label_columns[1:] + ['geometry']]

    df_cell_filtered_melted = df_cell_filtered.melt(id_vars='geometry', value_vars = best_cell_label_columns[1:], 
                        var_name='cells_to_check', value_name='value')
    df_cell_filtered_melted = df_cell_filtered_melted.dropna()
    df_cell_filtered_melted = pd.concat([df_cell_filtered_melted['value'], df_cell_filtered_melted['geometry']], axis=1)
    df_cell_filtered_melted.columns = ['cells_to_check', 'geometry']

    df1 = df_cell_filtered_melted.copy()
    df2 = gdf_cell_locations.copy()
    # Inner join the DataFrames based on 'Cell'
    merged_df = df1.merge(df2, left_on='cells_to_check', right_on='Cell', suffixes=('_df1', '_df2'))
    # Calculate the distance between points and create a list of distances
    distances = []
    for index, row in merged_df.iterrows():
        distance = geodesic((row['geometry_df1'].y, row['geometry_df1'].x), (row['geometry_df2'].y, row['geometry_df2'].x)).kilometers
        distances.append(distance)
    # Assign the list of distances to a new column
    merged_df['distance'] = distances

    min_distance_rows = merged_df.groupby('cells_to_check')['distance'].idxmin()
    # Create a new DataFrame with the rows that have the smallest distance
    df_cells_to_remove = merged_df.loc[min_distance_rows]
    df_cells_to_remove_return = df_cells_to_remove.copy()
    df_cells_to_remove = df_cells_to_remove[df_cells_to_remove['distance'] > 3]
    cell_to_remove = list(set(df_cells_to_remove['cells_to_check']))
    return cell_to_remove, df_cell_counts_return, df_cells_to_remove_return

def filter_cell_based_on_zero_columns(df_cell_occurrences):
    df = df_cell_occurrences
    df['zero_count'] = df.apply(lambda row: (row == 0).sum(), axis=1)
    df = df[df['zero_count'] >= 4]
    cells_to_remove = list(set(df['Cell']))
    return cells_to_remove

def filter_top_cells(df_segment, gdf_cell_locations):
    result = filter_cell_based_on_distance(df_segment, gdf_cell_locations)
    cells_to_remove1 = result[0]
    cells_to_remove2 = filter_cell_based_on_zero_columns(result[1])
    cells_to_remove = set(cells_to_remove1 + cells_to_remove2)
    cells_to_remove = [item for item in cells_to_remove if item[-2] != 'C']
    return cells_to_remove

def save_cluster_outline_kml(boundary_gdf, output):
    kml = Kml()
    for idx, row in boundary_gdf.iterrows():
        polygon = row['geometry']
        # Create a KML polygon with a red outline and no fill color
        kml_polygon = kml.newpolygon(name=row['Name'])
        kml_polygon.outerboundaryis = list(polygon.exterior.coords)
        kml_polygon.description = row['Description']
        color = (255,255,255,128)
        outline_color = ((255, 19, 252, 255))
        kml_polygon.style.linestyle.color = Color.rgb(*outline_color)
        kml_polygon.style.linestyle.width = 2
        kml_polygon.style.polystyle.color = Color.rgb(*color)
    kml.savekmz(f"{output}.kmz")

def cluster(df,area_name,date_str,output_file, threshold =1.33, metric='pilot_polution', eps=35, min_samples=5):
    '''
    threshold: pilot = 1.33, rsrp = -108
    metric: "pilot_polution" or "Median RSRP"
    '''
    df = df.copy()
    df['polygon_str'] = df['polygon_str'].str.replace('POLYGON ', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df['geometry'] = df['polygon_str'].apply(lambda x: Polygon([tuple(map(float, c.split())) for c in x.split(',')]))
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = "epsg:4326"
    gdf = gdf.to_crs('EPSG:32648')
    gdf['center_point'] = gdf['geometry'].centroid
    gdf['latitude'] = gdf['center_point'].y
    gdf['longitude'] = gdf['center_point'].x
    bins = [-float('inf'), threshold, float('inf')]
    gdf[f"{metric}_cut"] = pd.cut(gdf[metric], bins=bins, labels=False)
    if metric == 'Median RSRP':
        gdf = gdf[gdf[f"{metric}_cut"] == 0]
    if metric == 'pilot_polution':
        gdf = gdf[gdf[f"{metric}_cut"] == 1]
    
    if not gdf.empty:
        ## DBSCAN
        X = gdf[['longitude', 'latitude']]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        gdf['cluster'] = labels
        ## Takes instances that belong to a cluster
        gdf = gdf[gdf['cluster'] != -1][['geometry','cluster']].to_crs('EPSG:4326')

        ## Create a Multipolygon for each cluster
        grouped_gdf = gdf.groupby('cluster',group_keys=True)['geometry'].apply(lambda x: MultiPolygon(list(x))).reset_index()
        multi_polygon_gdf = gpd.GeoDataFrame(grouped_gdf, geometry='geometry', crs=gdf.crs)
        multi_polygon_gdf = multi_polygon_gdf.rename(columns={'cluster': 'cluster_id'})
        ## Create a polygon with the boundary of each multipolygon (each cluster)
        boundary_polygons = []
        for index, row in multi_polygon_gdf.iterrows():
            geometry = row['geometry']
            boundary = geometry.envelope
            boundary_polygons.append(boundary)
        boundary_gdf = gpd.GeoDataFrame({'geometry': boundary_polygons}, crs=multi_polygon_gdf.crs)

        ## Conbine polygons that have the intersections to one single polygon and save to a kml file
        ### Find the intersections between polygons and create a dictionary to track groups
        intersection_dict = {}
        group_id = 0
        for idx, polygon in boundary_gdf.iterrows():
            intersects = False
            for key, group in intersection_dict.items():
                if polygon['geometry'].intersects(group):
                    intersection_dict[key] = unary_union([group, polygon['geometry']])
                    intersects = True
                    break
            # If the current polygon doesn't intersect with any existing group, create a new group
            if not intersects:
                intersection_dict[group_id] = polygon['geometry']
                group_id += 1

        # Create a GeoDataFrame from the grouped polygons
        grouped_gdf = gpd.GeoDataFrame({'geometry': list(intersection_dict.values())}, crs=boundary_gdf.crs)
        a = "RSRP" if metric=='Median RSRP' else "Pilot"
        def get_cluster_name(row):
            return f"{area_name}_{a}_{date_str}_{row.name}"
        def get_cluster_description(row):
            return f"ID: {area_name}_{a}_{date_str}_{row.name}"
        grouped_gdf['Name'] = grouped_gdf.apply(get_cluster_name, axis=1)
        grouped_gdf['Description'] = grouped_gdf.apply(get_cluster_description, axis=1)
        grouped_gdf
        save_cluster_outline_kml(grouped_gdf, f"{output_file}")
    else:
        print(f"==={metric}--> an empty dataframe ===")


def generate_pilot_output(area_name, date_str, polygon_df, bin_size, df_raw, output_dir):
    '''
    '''
    print(f"***===RUNNING:{area_name}, {bin_size}*{bin_size} grid ==***")
    ### Read a Polygon and divide
    polygon_df = polygon_df.to_crs('EPSG:3857')
    bins_polygons = [generate_bins(geom, bin_size) for geom in polygon_df['geometry']]
    bins_polygons = [bin_poly for sublist in bins_polygons for bin_poly in sublist]
    # Create the GeoDataFrame for bins
    bins_df = gpd.GeoDataFrame({'geometry': bins_polygons}, crs=polygon_df.crs)
    bins_df['centroid'] = bins_df['geometry'].centroid
    bins_df_4326 = bins_df.to_crs("EPSG:4326")
    bins_df_4326['centroid'] = bins_df_4326['centroid'].to_crs("EPSG:4326")
    bins_df_4326['longitude'] = bins_df_4326['centroid'].x
    bins_df_4326['latitude'] = bins_df_4326['centroid'].y
    bins_df_4326['bin_id'] = "bin" + "_" +bins_df_4326['latitude'].astype(str)+ "_" + bins_df_4326['longitude'].astype(str)
    bins_df_4326 = bins_df_4326[['bin_id', 'geometry']]
    print('==Complete dividing bins')
    ### Load raw data file
    df_point = df_raw.copy()
    df_point_columns = [
    'Start Time',
    'eNodeB',
    'EARFCN (DL)',
    'EARFCN (UL)',
    'Physical Cell ID',
    'Latitude',
    'Longitude',
    'UL Volume (kB)',
    'DL Volume (kB)',
    'CQI 0',
    'CQI 1',
    'CQI 2',
    'CQI 3',
    'CQI 4',
    'CQI 5',
    'CQI 6',
    'CQI 7',
    'CQI 8',
    'CQI 9',
    'CQI 10',
    'CQI 11',
    'CQI 12',
    'CQI 13',
    'CQI 14',
    'CQI 15',
    'Serving Cell Label',
    'Serving Cell RSRP',
    'Best Cell Label',
    'Best Cell RSRP',
    'Second Best Cell Label',
    'Second Best Cell RSRP',
    'Third Best Cell Label',
    'Third Best Cell RSRP',
    'Fourth Best Cell Label',
    'Fourth Best Cell RSRP',
    'Fifth Best Cell Label',
    'Fifth Best Cell RSRP',
    'Sixth Best Cell Label',
    'Sixth Best Cell RSRP',
    ]
    df_point = df_point[df_point_columns]

    ### Locate bins for each sample
    bins_df_4326.reset_index(inplace=True)
    polygonDF = bins_df_4326.copy()
    polygonDF = polygonDF[['bin_id','geometry']]
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
    df_segment = df_segment.dropna(subset='Best Cell Label')
    df_segment['polygon_str'] = df_segment['polygon'].astype(str)


    ### Aggregation
    # Kml RSRP từng khu vực (tổ chức theo bin)#(tính Median best RSRP các segment trong bin)
    df_rsrp = df_segment[["polygon_str","bin_id", "Best Cell RSRP"]]
    df_rsrp_median = df_rsrp.groupby(['polygon_str','bin_id']).median()
    df_rsrp_count = df_rsrp.groupby(['polygon_str','bin_id']).count()
    df_rsrp = df_rsrp_median.rename(columns={'Best Cell RSRP': 'Median RSRP'})
    df_rsrp['No.Samples'] = df_rsrp_count['Best Cell RSRP']
    df_rsrp.reset_index(inplace=True)
    output_MedianRsrp_kml = os.path.join(output_dir, area_name+'_RSRP_'+str(bin_size)+'_'+date_str)
    save_output_to_kml(df_rsrp, 'Median RSRP', 'rsrp', output_MedianRsrp_kml)
    output_MedianRsrp_cluster_kml = os.path.join(output_dir, area_name+'_RSRP_Cluster_'+str(bin_size)+'_'+date_str)
    cluster(df_rsrp,area_name, date_str,output_MedianRsrp_cluster_kml,threshold=-108, metric='Median RSRP', eps=35, min_samples=5)
    print('==Complete Median RSRP')

    # Pilot Pollution
    pilot_columns = [
        'polygon_str',
        'bin_id',
        'Serving Cell Label',
        'Serving Cell RSRP',
        'Best Cell Label',
        'Best Cell RSRP',
        'Second Best Cell Label',
        'Second Best Cell RSRP',
        'Third Best Cell Label',
        'Third Best Cell RSRP',
        'Fourth Best Cell Label',
        'Fourth Best Cell RSRP',
        'Fifth Best Cell Label',
        'Fifth Best Cell RSRP',
        'Sixth Best Cell Label',
        'Sixth Best Cell RSRP'
    ]
    df_pilot = df_segment[pilot_columns].copy()
    df_pilot['rsrp_margin'] = df_pilot['Best Cell RSRP'] - 5
    n_bestcell_clumns = [
        'rsrp_margin',
        'Second Best Cell RSRP',
        'Third Best Cell RSRP',
        'Fourth Best Cell RSRP',
        'Fifth Best Cell RSRP',
        'Sixth Best Cell RSRP'
    ]
    compare_column = 'rsrp_margin'
    ## Code 1
    # df_pilot['pilot_polution'] = df_pilot[n_bestcell_clumns].apply(lambda row: sum(row[1:] > row[compare_column]), axis=1)
    ## Code 2, faster
    data_array = df_pilot[n_bestcell_clumns].values
    compare_array = df_pilot[compare_column].values
    counts = np.sum(data_array[:, 1:] > compare_array[:, np.newaxis], axis=1)
    df_pilot['pilot_polution'] = counts
    ## End of Code 2 ##
    df_pilot_agg = df_pilot[['polygon_str','bin_id', 'pilot_polution']].copy()
    df_pilot_agg_mean = df_pilot_agg.groupby(['polygon_str','bin_id']).mean()
    df_pilot_agg_count = df_pilot_agg.groupby(['polygon_str','bin_id']).count()
    df_pilot_agg = df_pilot_agg_mean
    df_pilot_agg['No.Samples'] = df_pilot_agg_count['pilot_polution']
    df_pilot_agg.reset_index(inplace=True)
    output_Pilot_kml = os.path.join(output_dir, area_name+'_Pilot_'+str(bin_size)+'_'+date_str)
    save_output_to_kml(df_pilot_agg, 'pilot_polution', 'pilot_pollution', output_Pilot_kml)
    output_Pilot_cluster_kml = os.path.join(output_dir, area_name+'_Pilot_Cluster_'+str(bin_size)+'_'+date_str)
    cluster(df_pilot_agg[df_pilot_agg['No.Samples'] >= 3],area_name, date_str,output_Pilot_cluster_kml,threshold=1.33, metric='pilot_polution', eps=35, min_samples=5)
    print('==Complete Pilot Pollution')

    #=================================FILTER CELLS BASED ON DISTANCE AND OCCURRENCES =================================
    df_cell_locations = pd.read_csv("../Raw/cell_locations.csv")
    geometry = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(df_cell_locations['Lon'], df_cell_locations['Lat'])])
    gdf_cell_locations = gpd.GeoDataFrame(df_cell_locations, geometry=geometry)[['Cell','geometry']]
    cells_to_remove = filter_top_cells(df_segment, gdf_cell_locations)
    #================================= Kml TOP 10 strongest cell trong bin <br>
    #================================= Ranking (Median RSRP từng best cell trong bin)*50% + Ranking (tần suất từng best cell trong bin)*50%  <br>
    #================================= => sort lấy top 10 cell
    grouped = df_segment.groupby(['polygon_str','bin_id','Best Cell Label'])
    aggregated = grouped.agg({'Best Cell RSRP': ['median', 'count']})
    aggregated.columns = ['median_rsrp', 'count']
    aggregated.reset_index(inplace=True)
    aggregated['median_rank'] = aggregated.groupby('polygon_str')['median_rsrp'].rank(method='dense')
    aggregated['count_rank'] = aggregated.groupby('polygon_str')['count'].rank(method='dense')
    aggregated['rank'] = (aggregated['median_rank'] + aggregated['count_rank']) / 2
    top_ranked_indices = aggregated.groupby('polygon_str')['rank'].nlargest(10).index.get_level_values(1)
    df3 = aggregated.loc[top_ranked_indices]
    df3 = df3[~df3['Best Cell Label'].isin(cells_to_remove)]
    # Create a custom description for each polygon
    df3 = df3.reset_index(drop=True)
    df3['description'] = df3.apply(lambda row: 
                                {
                                    'bin_id': row['bin_id'],
                                    'cell':row['Best Cell Label'],
                                    'medianRsrp':row['median_rsrp'],
                                    'medianRsrpRank':row['median_rank'],
                                    'freq': row['count'],
                                    'freqRank': row['count_rank'],
                                    'totalRank': row['rank']
                                    }, axis=1)
    # Group by 'polygon_str' and aggregate 'description' column to form a list
    df3_grouped = df3.groupby('polygon_str')['description'].apply(list).reset_index()
    df = df3_grouped.copy()
    df['polygon_str'] = df['polygon_str'].str.replace('POLYGON ', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df['geometry'] = df['polygon_str'].apply(lambda x: Polygon([tuple(map(float, c.split())) for c in x.split(',')]))
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    kml = Kml()
    for idx, row in gdf.iterrows():
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
                    <th>Bin_ID</th>
                    <th>Cell</th>
                    <th>Median RSRP</th>
                    <th>Median RSRP Rank</th>
                    <th>No.Samples</th>
                    <th>No.Samples Rank</th>
                    <th>Total Rank</th>
                </tr>
        """
        cell_list = row[['description']].item() # cell_list is a list of dict
        for item in cell_list:
            # item is a dict
            table_html += '<tr>'
            for key in item:
                table_html += f'<td>{item[key]}</td>'
            table_html += '<tr>'
        table_html += '</table>'   
        polygon = kml.newpolygon(name=str(idx), outerboundaryis=row['geometry'].exterior.coords[:])
        polygon.description = table_html
        polygon.style.linestyle.width = 1  # Set the width of the line
        polygon.style.linestyle.color = 'ff000000'  # Black
        polygon.style.polystyle.color = Color.rgb(0, 0, 0, 0)
        polygon.visibility = 0
    output_TopCells_kml = os.path.join(output_dir, area_name+'_TopCells_'+str(bin_size)+'_'+date_str+'.kmz')
    kml.savekmz(output_TopCells_kml)
    print('=====Complete Top Cells for each Bin=====')

    ##================================= Kml RSRP từng cell (tổ chức theo bin), mỗi cell 1 file KML <br>==========
    ##================================= (tính Median RSRP từng cell các segment trong bin)==========
    top_cell_list = list(set(df3['Best Cell Label']))
    top_cell_list.sort()
    df4 = df_segment[['bin_id','polygon_str','Best Cell Label','Best Cell RSRP']]
    df4 = df4[df4['Best Cell Label'].isin(top_cell_list)]
    df4_median = df4.groupby(['Best Cell Label','polygon_str','bin_id']).median()
    df4_count = df4.groupby(['Best Cell Label','polygon_str','bin_id']).count()
    df4 = df4_median
    df4['No.Samples'] = df4_count['Best Cell RSRP']
    cell_kml_dir = f"{output_dir}/Cell/"
    if not os.path.exists(cell_kml_dir):
        os.makedirs(cell_kml_dir)
        print(f"Directory '{output_dir}' created.")
    for cell, subset in df4.groupby(level='Best Cell Label'):
        subset = subset.reset_index()
        # display(subset)
        subset = subset[['polygon_str','Best Cell RSRP', 'No.Samples','bin_id']]
        output_cell_kml = os.path.join(cell_kml_dir, cell+'_'+str(bin_size)+'_'+date_str)
        save_output_to_kml(subset, 'Best Cell RSRP','rsrp', output_cell_kml)

    ## ================================   Store cell rsrp to 1 .kml file.
    kml = Kml()
    for cell in top_cell_list:
        network_link1 = kml.newnetworklink(name=cell)
        output_cell_kml = os.path.join(cell_kml_dir, cell+'_'+str(bin_size)+'_'+date_str)
        # network_link1.link.href = os.path.abspath(f"{output_cell_kml}.kml")
        network_link1.link.href = f"./Cell/{cell}_{str(bin_size)}_{date_str}.kmz"
    output_each_cell_kml = os.path.join(output_dir, area_name+'_RSRP_Footprint_'+str(bin_size)+'_'+date_str+'.kmz')
    kml.savekmz(output_each_cell_kml)
    print('=====Complete Median RSRP for each Cell=====')

    ##================================== EACH CELL ONE COLOR =========================
    df5 = aggregated.loc[aggregated.groupby(['polygon_str','bin_id'])['rank'].idxmax()]
    df = df5.reset_index(drop=True)
    ## Generate a random color for each cell
    import random
    unique_cells = df['Best Cell Label'].unique()
    color_mapping = {cell: '#' + ''.join(random.choices('0123456789ABCDEF', k=6)) for cell in unique_cells}
    df['color'] = df['Best Cell Label'].map(color_mapping)
    df['polygon_str'] = df['polygon_str'].str.replace('POLYGON ', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df['geometry'] = df['polygon_str'].apply(lambda x: Polygon([tuple(map(float, c.split())) for c in x.split(',')]))
    rgba_colors = [hex_to_rgba(color, alpha=int(0.8*256)) for color in df['color']]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    kml = Kml()
    for idx, row in gdf.iterrows():
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
                    <th>Bin_ID</th>
                    <th>Cell</th>
                    <th>No.Samples</th>
                </tr>
        """
        bin_id = row['bin_id']
        cell_label = row['Best Cell Label']
        count = row['count']
        table_html += f'<tr><td>{bin_id}</td><td>{cell_label}</td><td>{count}</td></tr>'
        table_html += '</table>'

        polygon = kml.newpolygon(name=str(idx), outerboundaryis=row['geometry'].exterior.coords[:])
        # polygon.description = row['Best Cell Label']
        polygon.description = table_html
        color = rgba_colors[idx]
        polygon.style.polystyle.color = Color.rgb(*color)
        polygon.style.linestyle.width = 1
        polygon.style.linestyle.color = Color.rgb(*color)
        polygon.visibility = 0
    output_BestServingCells_kml = os.path.join(output_dir, area_name+'_BestServingCells_'+str(bin_size)+'_'+date_str+'.kmz')
    kml.savekmz(output_BestServingCells_kml)
    print('==Complete Define Colors for Best Serving Cells')

    # ## Volume Data (UL+DL)
    # df5 = df_segment[['polygon_str','UL Volume (kB)','DL Volume (kB)']].copy()
    # df5['UL Volume (kB)'] = df5['UL Volume (kB)'].str.replace(',', '').astype(float)
    # df5['DL Volume (kB)'] = df5['DL Volume (kB)'].str.replace(',', '').astype(float)
    # df5['Volume'] = (df5['UL Volume (kB)'] + df5['DL Volume (kB)'])/1000
    # df5 = df5[['polygon_str', 'Volume']]
    # df5_sum = df5.groupby(['polygon_str']).sum()
    # df5_count = df5.groupby(['polygon_str']).count()
    # df5 = df5_sum
    # df5['No.Samples'] = df5_count['Volume']
    # df5.reset_index(inplace=True)
    # output_volume_kml = os.path.join(output_dir, area_name+'_Volume_'+str(bin_size)+'_'+date_str)
    # save_volume_data(df5, output_volume_kml)
    # print('==Complete Saving Volume Data')
    # ###
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"===Process completed: {area_name} , {bin_size}*{bin_size} ===")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"*****===== PROCESS COMPLETED: {area_name}, {bin_size}*{bin_size} grid =====*****")
    