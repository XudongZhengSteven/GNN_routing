# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rasterio
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from general_info import *
import numpy as np

from easy_vic_build.tools.plot_func.plot_utilities import set_boundary, set_xyticks, get_NDVI_cmap, get_colorbar, get_UMD_LULC_cmap
from easy_vic_build.tools.plot_func.plot_map import soft_hillshade, light_gray_dem, plot_Basin_map
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build.tools.dpc_func.basin_grid_func import createArray_from_gridshp, createStand_grids_lat_lon_from_gridshp

from HRB_build_dpc import dataProcess_VIC_level0_HRB, dataProcess_VIC_level1_HRB, dataProcess_VIC_level2_CMFD_HRB, dataProcess_VIC_level3_HRB
plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12

def plot_basin_map_combine(
    evb_dir_hydroanalysis,
    evb_dir_modeling,
    figsize=(7, 8),
    grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.1, "wspace": 0.08},
    x_locator_interval_landsurface=1, y_locator_interval_landsurface=0.8,
    x_locator_interval_grid=1, y_locator_interval_grid=0.8
):
    # read
    dpc_VIC_level0 = readdpc(evb_dir_modeling.dpc_VIC_level0_path, dataProcess_VIC_level0_HRB)
    dpc_VIC_level1 = readdpc(evb_dir_modeling.dpc_VIC_level1_path, dataProcess_VIC_level1_HRB)
    dpc_VIC_level3 = readdpc(evb_dir_modeling.dpc_VIC_level3_path, dataProcess_VIC_level3_HRB)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0]
    
    dpc_VIC_level1.merge_grid_data()
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0]
    
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"clipped_stream_vector_basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp"
    ))
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    gauge_lon = gauge_info[station_name]["gauge_coord(lon, lat)_level0"][0]
    gauge_lat = gauge_info[station_name]["gauge_coord(lon, lat)_level0"][1]
    
    # plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, **grid_kwarg)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    
    # plot dem
    grid_shp_level0.plot(ax=ax1, column="ASTGTM_DEM_mean_Value", alpha=1, legend=False, colormap="terrain", zorder=1,
                                 legend_kwds={"label": "Elevation (m)"})  # terrain gray
    # grid_shp_level0.plot(ax=ax2, facecolor="none", linewidth=0.1, alpha=1, edgecolor="k", zorder=2)
    # grid_shp_level0.plot(ax=ax2, facecolor="k", alpha=0.2, zorder=3)
    stream_gdf.plot(ax=ax1, color="b", zorder=4)

    ax1.plot(gauge_lon, gauge_lat, "r^", markersize=8, mec="k", mew=1, zorder=5)
    basin_shp.plot(ax=ax1, edgecolor="k", alpha=1, facecolor="none", zorder=4)
    # fig, ax2 = dpc_VIC_level1.plot(fig, ax2, basin_shp_kwargs={"edgecolor": "k", "alpha": 0.1, "facecolor": "b"})  # grid
    set_boundary(ax1, grid_shp_level0.createBoundaryShp()[-1])
    set_xyticks(ax1, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot grid
    basin_shp.plot(ax=ax2, edgecolor="k", alpha=0.5, facecolor="b")
    grid_shp_level1.plot(ax=ax2, alpha=0.5, edgecolor="k", facecolor="none", linewidth=0.5)
    grid_shp_level1.point_geometry.plot(ax=ax2, alpha=0.5, color="darkblue", markersize=1)        
    # fig, ax3 = grid_shp_level1.plot(fig, ax3)
    set_boundary(ax2, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax2, x_locator_interval=x_locator_interval_grid, y_locator_interval=y_locator_interval_grid, yticks_rotation=90)
    
    # plot LULC
    UMD_LULC_cmap, UMD_LULC_norm, UMD_LULC_ticks, UMD_LULC_ticks_position, UMD_LULC_colorlist, UMD_LULC_colorlevel = get_UMD_LULC_cmap()
    grid_shp_level1.plot(ax=ax3, column="umd_lc_major_Value", alpha=1, legend=False, colormap=UMD_LULC_cmap, zorder=1, norm=UMD_LULC_norm,
                                 legend_kwds={"label": "UMD LULC"})  # terrain gray
    set_boundary(ax3, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax3, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot Veg
    ndvi_cmap = get_NDVI_cmap()
    grid_shp_level1["MODIS_NDVI_mean_Value_month7_scaled"] = grid_shp_level1["MODIS_NDVI_mean_Value_month7"] * 0.0001 * 0.0001
    grid_shp_level1.plot(ax=ax4, column="MODIS_NDVI_mean_Value_month7_scaled", alpha=1, legend=False, colormap=ndvi_cmap, zorder=1,
                                 legend_kwds={"label": "NDVI"}, vmin=0, vmax=1)  # Greens
    set_boundary(ax4, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax4, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # ------------ plot colorbar ------------
    # dem cb
    dem_values = grid_shp_level0["ASTGTM_DEM_mean_Value"].values
    dem_vmin = dem_values.min()
    dem_vmax = dem_values.max()
    dem_cmap = "terrain"
    fig_dem_cb, ax_dem_cb, _, _ = get_colorbar(dem_vmin, dem_vmax, dem_cmap, figsize=(4, 2), subplots_adjust={"right": 0.5}, cb_label="", cb_label_kwargs={}, cb_kwargs={"orientation":"vertical"})
    
    # lulc cb
    lulc_vmin = -0.5
    lulc_vmax = 13.5
    lulc_cmap = UMD_LULC_cmap
    fig_lulc_cb, ax_lulc_cb, _, _ = get_colorbar(lulc_vmin, lulc_vmax, lulc_cmap, figsize=(6, 1), subplots_adjust={"bottom": 0.5}, cb_label="UMD LULC Classification", cb_label_kwargs={}, cb_kwargs={"orientation":"horizontal", "ticks": UMD_LULC_ticks_position})
    
    # NDVI cb
    ndvi_vmin = 0
    ndvi_vmax = 1
    ndvi_cmap = ndvi_cmap
    fig_ndvi_cb, ax_ndvi_cb, _, _ = get_colorbar(ndvi_vmin, ndvi_vmax, ndvi_cmap, figsize=(6, 1), subplots_adjust={"bottom": 0.5}, cb_label="NDVI", cb_label_kwargs={}, cb_kwargs={"orientation":"horizontal"})
    
    # ------------ save fig ------------
    fig.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_Basin_map_combine.tiff"), dpi=300)
    fig_dem_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_dem_cb.svg"), dpi=300)
    fig_lulc_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_lulc_cb.svg"), dpi=300)
    fig_ndvi_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_ndvi_cb.svg"), dpi=300)
    
    
def plot_basin_map_JRB(evb_dir_hydroanalysis, evb_dir_modeling):
    # read dpc_VIC_level1
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_HRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"))
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(evb_dir_modeling._dpc_VIC_level3_path)
    
    # read stream gdf
    station_id = basin_outlets_reference_i_map[station_name]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"clipped_stream_vector_basin_vector_outlet_with_reference_{station_id}.shp"
    ))
    
    # plot
    fig_dict, ax_dict = plot_Basin_map(
        dpc_VIC_level0,
        dpc_VIC_level1,
        dpc_VIC_level2_CMFD,
        stream_gdf,
        dpc_VIC_level3.get_data_from_cache("gauge_info")[0][station_name]["gauge_coord(lon, lat)_level0"],
        x_locator_interval=1, y_locator_interval=0.5,
        fig=None, ax=None,
        dem_column="ASTGTM_DEM_mean_Value",
    )
    
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level0"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level0.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level1"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level1.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level2"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level2.tiff"), dpi=300)

def plot_basin_map_paper(evb_dir_hydroanalysis, evb_dir_modeling):
    # read
    dpc_VIC_level0 = readdpc(evb_dir_modeling.dpc_VIC_level0_path, dataProcess_VIC_level0_HRB)
    dpc_VIC_level1 = readdpc(evb_dir_modeling.dpc_VIC_level1_path, dataProcess_VIC_level1_HRB)
    dpc_VIC_level3 = readdpc(evb_dir_modeling.dpc_VIC_level3_path, dataProcess_VIC_level3_HRB)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0]
    
    dpc_VIC_level1.merge_grid_data()
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0]
    
    # read shp
    basin_shps = dpc_VIC_level3.get_data_from_cache("basin_shps")[0]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"clipped_stream_vector_basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp"
    ))
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    gauge_lon = {}
    gauge_lat = {}
    
    for name in station_names:
        gauge_lon[name] = gauge_info[name]["gauge_coord(lon, lat)_level0"][0]
        gauge_lat[name] = gauge_info[name]["gauge_coord(lon, lat)_level0"][1]


    # read dem
    dem_path = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip_basin.tif"
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    dem[dem == -9999] = np.nan
    
    # get boundary
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level0, reverse_lat
    )
    
    xmin = np.min(stand_grids_lon)
    xmax = np.max(stand_grids_lon)
    ymin = np.min(stand_grids_lat)
    ymax = np.max(stand_grids_lat)
    
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax = axes[0]
    ax.imshow(
        dem,
        cmap="BrBG",# "terrain", #"BrBG",
        extent=extent,
        origin="upper",
        alpha=0.8,
    )
    
    for i, name in enumerate(station_names):
        basin_shps[name].plot(
            ax=ax,
            edgecolor="k",
            alpha=0.8,
            facecolor="none",
            zorder=4,
            linewidth=0.8,
        )
    
    stream_gdf.plot(
        ax=ax,
        edgecolor="mediumblue",
        alpha=0.8,
        facecolor="none",
        zorder=3,
        linewidth=0.8,
    )
    
    ax.scatter(
        [gauge_lon[name] for name in station_names],
        [gauge_lat[name] for name in station_names],
        s=30,
        marker="^",
        facecolor="red",
        edgecolor="black",
        linewidth=1.0,
        zorder=12,
    )
    
    set_boundary(ax, [xmin, ymin, xmax, ymax])
    set_xyticks(ax, x_locator_interval=0.5, y_locator_interval=0.5, yticks_rotation=90)
    # plt.colorbar(ax.images[0], ax=ax, label="Elevation (m)")
    
    cbar = plt.colorbar(ax.images[0], ax=ax, label="Elevation (m)")
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height * 0.8])
    
    plt.show(block=True)
    
    # plot location and grids
    ax2 = axes[1]

def plot_lulc_analysis():
    lulc_fp = r"F:\research\Research\ModelingUncertainty_hanjiang\data\LULC\clipped_CNLUCC.tif"
    lulc_dataset = rasterio.open(lulc_fp, mode="r")
    lulc_band = lulc_dataset.read(1)
    lulc_band_flatten = lulc_band.flatten()
    lulc_band_flatten_valid = lulc_band_flatten[lulc_band_flatten != lulc_dataset.nodata]
    
    unique, counts = np.unique(lulc_band_flatten_valid, return_counts=True)

    df = pd.DataFrame({
        "class_id": unique,
        "count": counts
    })

    df["percentage"] = df["count"] / df["count"].sum() * 100

    # =========================
    # 2. 细分类名称映射
    # =========================
    class_names = {
        11: "Irrigated paddy field",
        12: "Rainfed cropland",
        21: "Forested land",
        22: "Shrubland",
        23: "Sparse forest",
        24: "Other forest land",
        31: "High-cover grassland",
        32: "Medium-cover grassland",
        33: "Low-cover grassland",
        41: "River / Canal",
        42: "Lake",
        43: "Reservoir / Pond",
        45: "Tidal flat",
        46: "Floodplain",
        51: "Urban area",
        52: "Rural residential",
        53: "Other built-up land",
        61: "Sandy land",
        63: "Saline land",
        64: "Wetland",
        65: "Bare land",
        66: "Bare rock"
    }

    df["class_name"] = df["class_id"].map(class_names)

    # =========================
    # 3. 大类分类函数
    # =========================
    def classify_major(cat):
        if cat in [11, 12]:
            return "Cropland"
        elif cat in [21, 22, 23, 24]:
            return "Forest"
        elif cat in [31, 32, 33]:
            return "Grassland"
        elif cat in [41, 42, 43, 45, 46]:
            return "Water bodies"
        elif cat in [51, 52, 53]:
            return "Urban / Built-up"
        else:
            return "Unused land"

    df["major_class"] = df["class_id"].apply(classify_major)

    # =========================
    # 4. Major统计
    # =========================
    df_major = df.groupby("major_class")["count"].sum().reset_index()
    df_major["percentage"] = df_major["count"] / df_major["count"].sum() * 100

    # =========================
    # 5. Anthropogenic vs Natural
    # =========================
    def anthro_vs_natural(cat):
        if cat in [11, 12, 51, 52, 53]:
            return "Anthropogenic"
        else:
            return "Natural"

    df["group"] = df["class_id"].apply(anthro_vs_natural)

    df_group = df.groupby("group")["count"].sum().reset_index()
    df_group["percentage"] = df_group["count"] / df_group["count"].sum() * 100

    # 顺序固定（论文习惯）
    df_group = df_group.set_index("group").loc[["Natural", "Anthropogenic"]].reset_index()

    # =========================
    # 6. 配色（统一语义）
    # =========================
    color_map = {
        "Cropland": "#FB8C00",
        "Forest": "#2E7D32",
        "Grassland": "#66BB6A",
        "Water bodies": "#1E88E5",
        "Urban / Built-up": "#D32F2F",
        "Unused land": "#9E9E9E"
    }

    pie_colors = {
        "Natural": "#4CAF50",
        "Anthropogenic": "#E64A19"
    }

    # =========================
    # 7. 排序
    # =========================
    df_sorted = df.sort_values(["major_class", "percentage"], ascending=[True, False])
    df_major_sorted = df_major.sort_values("percentage", ascending=False)

    # =========================
    # 8. 绘图（三联图）
    # =========================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -------- (a) 细分类（继承大类颜色） --------
    colors_a = [color_map[m] for m in df_sorted["major_class"]]

    axes[0].barh(
        df_sorted["class_name"],
        df_sorted["percentage"],
        color=colors_a
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Percentage (%)")

    # 分组分隔线（增强结构）
    major_changes = df_sorted["major_class"].ne(df_sorted["major_class"].shift())
    for i in np.where(major_changes)[0]:
        axes[0].axhline(i - 0.5, color='gray', linewidth=0.5)

    # -------- (b) Major classes --------
    colors_b = [color_map[m] for m in df_major_sorted["major_class"]]

    axes[1].bar(
        df_major_sorted["major_class"],
        df_major_sorted["percentage"],
        color=colors_b
    )
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis='x', rotation=30)
    
    # -------- (c) Anthropogenic vs Natural --------
    axes[2].pie(
        df_group["percentage"],
        labels=df_group["group"],
        autopct='%1.1f%%',
        colors=[pie_colors[g] for g in df_group["group"]],
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )

    # =========================
    # 9. 全局美化
    # =========================
    for ax in axes[:2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # legend（统一说明颜色）
    handles = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    # axes[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.savefig(os.path.join(r"F:\research\Research\ModelingUncertainty_hanjiang", "fig_LULC_analysis.tiff"), dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    # plot_basin_map_JRB(evb_dir_hydroanalysis, evb_dir_modeling)
    # plot_basin_map_combine(evb_dir_hydroanalysis, evb_dir_modeling)
    # plot_basin_map_paper(evb_dir_hydroanalysis, evb_dir_modeling)
    plot_lulc_analysis()
    