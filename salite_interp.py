import re
import os
import csv
import glob
import cv2
import tqdm
import pymysql
import warnings
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from logzero import logger
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
warnings.filterwarnings("ignore")



class CreateXarryInterp(ABC):
    def __init__(self, fdata, time_mode='utc2cst'):
        self.dr = fdata
        if time_mode == 'utc2cst':
            self._convert_utc2cst()

    @abstractmethod
    def create_interp_obj(self, lat_list, lon_list, name_list=None):
        pass

    def _utc2cst(self, time_item):
        return pd.to_datetime(str(time_item + np.timedelta64(8,'h')))

    def _convert_utc2cst(self):
        timeutc = self.dr.time.values[0]
        timecst = self._utc2cst(timeutc)
        self.dr['time_cst'] = timecst.strftime('%Y-%m-%d %H:%M:%S')
        # self.dr['time_cst_10T'] = timecst.round(freq = '10T').strftime('%Y-%m-%d %H:%M:%S')

    def interp_values(self, lat, lon):
        #logger.debug(f'interp_values_dr:{self.dr}')
        interped_dr = self.dr.interp(
            # latitude = lat,
            # longitude = lon
            lat = lat,
            lon = lon
        )
        return interped_dr

    def convert_to_df(self, dr):
        df = dr.to_dataframe()
        return df.reset_index()


class StationInterp(CreateXarryInterp):
    def __init__(self, fdata,time_mode='utc2cst'):
        super().__init__(fdata, time_mode)
    def create_interp_obj(self, lat_list, lon_list, name_list=None):
        station_lats = xr.DataArray(
            lat_list,
            # dims = "station",
            coords=[name_list],
            dims = "station"
        )
        station_lons = xr.DataArray(
            lon_list,
            coords=[name_list],
            dims="station"
        )
        return station_lats, station_lons


class SearchFile(object):

    def __init__(self, elementname):
        self.main_dir = r'/data2/project_data/HeightIsothermal/2021/*/*/*.nc'
        self.all_file_list = None

    def get_path_list(self, start_index, end_index):
        file_list = sorted(glob.glob(self.main_dir))
        #print(file_list[-10:])
        #raise ValueError
        self.all_file_list = file_list[start_index:end_index]

    def _extra_element_name_time(self, fpath):
        element_name = fpath.split('-')[-2][1:]
        pattern = r'[0-9]+'
        sytex = re.compile(pattern)
        time_str = re.findall(sytex, fpath)[-3]
        return element_name, time_str

    def iter_data(self, start_index, end_index):
        if self.all_file_list is None:
            self.get_path_list(start_index, end_index)
        for file in self.all_file_list:
            fdata = xr.open_dataset(file)
            yield fdata
            # element_name, time_str = self._extra_element_name_time(file)
            # yield fdata, element_name, time_str


def product_interp_1d(filename, lat, lon):
    element_name_list = ['POT','POH','POSW','POHR']
    with open('/home/testLZD/Fy4a/data/salite_interp_station.csv', 'a') as f:
        dr = xr.open_dataset(filename)
        sti = StationInterp(dr)
        station_table = pd.read_csv(r'/home/cqkj/project/YellowRiverBasin/verify/qh_cimiss_station_info.csv')
        lat_list, lon_list, name_list = station_table.Lat.values.squeeze(), station_table.Lon.values.squeeze(), station_table.Station_Id_C.values.squeeze()
        station_lat, station_lon = sti.create_interp_obj(lat_list, lon_list, name_list)
        #logger.debug(f"station_lat:{station_lat}, station_lon:{station_lon}")
        station_dr = sti.interp_values(station_lat,station_lon)
        sub_df = sti.convert_to_df(station_dr)
        logger.debug("="*20)
        station_df_sub = sub_df[['station', 'time_cst']+ element_name_list]
        logger.debug(station_df_sub)
        # logger.info("采用协程进行数据写入")
        # yield station_df_sub
        writer = csv.writer(f)
        for row in sub_df.values:
            writer.writerow(row)
            #writer.writerow(station_df_sub.values)
    logger.debug("csv写入成功！")






def run(file_list):
    logger.debug(f"len file_list is:{len(file_list)}, type is:{type(file_list)}")
    logger.info(file_list)
    try:
        interp_1d(file_list, "", "")
    except Exception as e:
        logger.error(e)

    #try:
    #    for file in file_list:
    #        interp_1d(file, "", "")
    #except Exception as e:
    #    logger.error(e)




def pool_run():
    fpath = r'/public/project_data/YellowRiver/nowcasting/PROB_sate_fcst_decide/Qinghai/*/*/*/*.nc'
    pool_size = 4
    file_list = sorted(glob.glob(fpath))[:4]
    logger.info(file_list[0])
    split_file = list(range(0, len(file_list), 100))
    begin, end = split_file[:-1], split_file[1:]
    with ProcessPoolExecutor(pool_size) as pool:
        # task = [pool.submit(run, file_list[b:e]) for b,e in zip(begin, end)]
        task = pool.map(run, file_list)


if __name__ == '__main__':
    pool_run()
