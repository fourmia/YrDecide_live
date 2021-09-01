# -*- coding: utf-8 -*-
"""
Created on 2021/5/26 23:44

@author: LXL
@description: 
"""
import numpy as np
import xarray as xr
import os
import glob
import pandas as pd
import traceback
import argparse
from datetime import datetime, timedelta
from scipy import ndimage
from skimage import morphology, measure, draw, filters
import argparse
import traceback

from dataHandler import sql_handler
from dataInterface import fileReader, fileManager
from collections import defaultdict
from typing import List
from verify import deal_observation

REGION = 'Qinghai'

SEP_DIR = "/public/project_data/YellowRiver/nowcasting/SEP"
PROB_DIR = '/public/project_data/YellowRiver/nowcasting/PROB_sate_fcst'
FY4A_DIR = "/public/project_data/SharedData/FY4A/Qinghai"


THR_ZOHR = 75
THR_ZOH = 85
THR_ZOT = 60
THR_ZOSW = 60

fileManager.CONFIG.update(dict(DIR_FY4A=FY4A_DIR))


def get_satellite(st, et, cinfo) -> dict:
    timespan = [st, et]
    lons = xr.DataArray(cinfo.lon.values, dims='cid')
    lats = xr.DataArray(cinfo.lat.values, dims='cid')
    data_time = xr.DataArray(cinfo.data_time.values, dims='cid')
    cid = cinfo.data_type.values
    source = 'FY4A'
    ptypes = ['CTT', 'CTP', 'CTH', 'QPE', 'FDI']
    data_dict = {}
    for ptype in ptypes:
        paths = fileManager.get_path(source, timespan=timespan, instrument='AGRI',
                                     ptype=ptype, suffix='.NC')

        dfs = []
        for path in paths:
            attrs = os.path.basename(path).split('.')[0].split('_')
            scan_st, scan_et = pd.to_datetime(attrs[15]), pd.to_datetime(attrs[16])
            cond = (data_time >= scan_st) & (data_time <= scan_et)
            sel_lons = lons[cond]
            sel_lats = lats[cond]
            sel_type = cid[cond.data]
            sel_dtime = data_time[cond]

            if len(sel_lons):
                nc = xr.open_dataset(path, decode_times=False)
                if ptype in ['CTT', 'CTP', 'CTH', 'QPE']:
                    if ptype == 'QPE':
                        nc = nc.rename({'Precipitation': 'QPE'})
                    data = nc[ptype].interp(lat=sel_lats, lon=sel_lons,
                                            method='nearest')
                    df = data.to_dataframe()
                    df.reset_index(inplace=True)
                    df['cid'] = sel_type
                    df['data_time'] = pd.to_datetime(sel_dtime.data)
                    dfs.append(df)

                elif ptype in ['FDI']:
                    c09 = nc["Channel09"].interp(lat=sel_lats, lon=sel_lons,
                                                 method='nearest')
                    c12 = nc["Channel12"].interp(lat=sel_lats, lon=sel_lons,
                                                 method='nearest')
                    df09 = c09.to_dataframe()
                    df09.reset_index(inplace=True)
                    df09.columns = ['time', 'cid', 'lat', 'lon', 'Channel09']
                    df12 = c12.to_dataframe()
                    df12.reset_index(inplace=True)
                    df12.columns = ['time', 'cid', 'lat', 'lon', 'Channel12']

                    df = pd.merge(df09, df12, on=['time', 'cid', 'lat', 'lon'])
                    df.reset_index(inplace=True)
                    df['cid'] = sel_type
                    df['data_time'] = pd.to_datetime(sel_dtime.data)
                    dfs.append(df)

                else:
                    df = None

        if len(dfs):
            con = pd.concat(dfs, axis=0, ignore_index=True)
            if ptype in ['FDI']:
                data_dict['Channel09'] = con.loc[:, ['cid', 'data_time', 'lat', 'lon', 'Channel09']]
                data_dict['Channel12'] = con.loc[:, ['cid', 'data_time', 'lat', 'lon', 'Channel12']]
            else:
                data_dict[ptype] = con.loc[:, ['cid', 'data_time', 'lat', 'lon', ptype]]

    keys = list(data_dict.keys())
    if len(keys) >= 2:
        datas = data_dict.get(keys[0])
        for k in keys[1:]:
            datas = pd.merge(datas, data_dict.get(k), on=['cid', 'data_time', 'lat', 'lon'])
            datas.drop_duplicates(inplace=True, ignore_index=True)

    elif len(keys) == 1:
        datas = data_dict.get(keys[0])

    else:
        datas = None

    return datas


class DynamicThreshold:
    """
    st, et 为世界时
    """

    def __init__(self, st, et):
        self.st = st
        self.et = et
        self.bjt_st = st + timedelta(hours=8)
        self.bjt_et = et + timedelta(hours=8)
        self.sinfo = deal_observation.SINFO
        self.data_hr = self.get_hr(self.bjt_st, self.bjt_et)
        self.data_li = self.get_li(self.bjt_st, self.bjt_et)
        self.data_station_li = self.get_satation_li(self.bjt_st, self.bjt_et)
        self.data_tg = self.get_tg(self.bjt_st, self.bjt_et)
        self.data_h = self.get_hail(self.bjt_st, self.bjt_et)
        self.cinfo = self.get_cinfo()
        self.data_sate = self.get_sate(self.st, self.et, self.cinfo)
        self.thrs = self.render_thr()

    def get_hr(self, st, et):
        hr = deal_observation.HeavyRain(st, et)
        hr.thr = 5
        data_hr = hr.identify()
        data_hr = data_hr[data_hr.flag_hr == 1]
        data_hr['data_time'] = pd.to_datetime(data_hr['data_time']) - timedelta(hours=8)

        return data_hr

    def get_li(self, st, et):
        data_li = deal_observation.Lightning.get_data(st, et)
        data_li['data_time'] = pd.to_datetime(data_li['data_time']) - timedelta(hours=8)

        return data_li

    def get_satation_li(self, st, et):
        data_li = deal_observation.Lightning(st, et).identify(self.sinfo, self.data_li)
        data_li = data_li[data_li.flag_li == 1]
        data_li['data_time'] = pd.to_datetime(data_li['data_time']) - timedelta(hours=8)

        return data_li

    def get_tg(self, st, et):
        data_tg = deal_observation.ThunderstormGale(st, et).identify(self.data_station_li)
        data_tg = data_tg[data_tg.flag_tg == 1]
        data_tg['data_time'] = pd.to_datetime(data_tg['data_time']) - timedelta(hours=8)

        return data_tg

    def get_hail(self, st, et):
        data_h = pd.DataFrame(columns=['cid', 'data_time', 'flag_h'])

        return data_h

    def get_cinfo(self):
        sinfo = self.sinfo.loc[:, ['Station_Id_C', 'Lon', 'Lat']]
        sinfo.columns = ['sid', 'lon', 'lat']
        cinfo_ = []
        if len(self.data_hr):
            hr = pd.merge(self.data_hr, sinfo, on=['sid'])
            hr = hr.loc[:, ['data_time', 'lon', 'lat']]
            hr['data_type'] = 'hr'
            cinfo_.append(hr)
        if len(self.data_tg):
            tg = pd.merge(self.data_tg, sinfo, on=['sid'])
            tg = tg.loc[:, ['data_time', 'lon', 'lat']]
            tg['data_type'] = 'tg'
            cinfo_.append(tg)
        if len(self.data_h):
            h = pd.merge(self.data_h, sinfo, on=['sid'])
            h = h.loc[:, ['data_time', 'lon', 'lat']]
            h['data_type'] = 'h'
            cinfo_.append(h)
        if len(self.data_li):
            li = self.data_li.loc[:, ['data_time', 'lon', 'lat']]
            li['data_type'] = 'li'
            cinfo_.append(li)
        cinfo = pd.concat(cinfo_, axis=0, ignore_index=True)
        cinfo.reset_index(drop=True, inplace=True)

        return cinfo

    def get_sate(self, st, et, cinfo):
        datas = get_satellite(st, et, cinfo)
        return datas

    def _calc_thr(self, data):
        ctt = data.CTT[data.CTT > 100]
        ctp = data.CTP[data.CTP > 10]
        cth = data.CTH[data.CTH > 3000]
        qpe = data.QPE[data.QPE > 0]
        vap = data.Channel09[data.Channel09 > 100]
        ir = data.Channel12[data.Channel12 > 100]
        q = [10, 90]
        thr_ctt = np.percentile(ctt, q=q) if len(ctt) else None
        thr_ctp = np.percentile(ctp, q=q) if len(ctp) else None
        thr_cth = np.percentile(cth, q=q) if len(cth) else None
        thr_qpe = np.percentile(qpe, q=q) if len(qpe) else None
        thr_vap = np.percentile(vap, q=q) if len(vap) else None
        thr_ir = np.percentile(ir, q=q) if len(ir) else None

        return thr_ctt, thr_ctp, thr_cth, thr_qpe, thr_vap, thr_ir

    def thr_hr(self):
        sel = self.data_sate[self.data_sate.cid == 'hr']
        if len(sel):
            thrs = self._calc_thr(sel)
            return thrs

    def thr_li(self):
        sel = self.data_sate[self.data_sate.cid == 'li']
        print(sel)
        if len(sel) >= 3:
            thrs = self._calc_thr(sel)
            return thrs

    def thr_tg(self):
        sel = self.data_sate[self.data_sate.cid == 'tg']
        if len(sel):
            thrs = self._calc_thr(sel)
            return thrs

    def thr_h(self):
        sel = self.data_sate[self.data_sate.cid == 'h']
        if len(sel):
            thrs = self._calc_thr(sel)
            return thrs

    def render_thr(self):
        if self.data_sate is not None:
            thrs = {'POHR': self.thr_hr(),
                    'POH': self.thr_h(),
                    'POT': self.thr_li(),
                    'POSW': self.thr_tg(),
                    }
        else:
            thrs = {}

        return thrs


def sigmoid(x, a=-0.1, b=40, reverse=False):
    """
    雷电：a=-0.1, b=40
    冰雹：a=-0.1, b=50
    雷暴大风：a=-0.1, b=55
    短时强降水：a=-0.1, b=14
    :param x:
    :param a:
    :param b:
    :return:
    """
    p = 100 * 1.0 / (1 + np.exp(a * (x - b)))
    if reverse:
        p = 100 - p

    return p


def show_sigmoid(a=-0.1, b=40, xmin=0, xmax=75, reverse=False):
    import matplotlib.pyplot as plt

    x = np.linspace(xmin, xmax, 100)
    y = sigmoid(x, a=a, b=b)
    if reverse:
        y = 100 - y
    plt.figure()
    plt.plot(x, y)


def zone_identify(x, thr=50):
    """
    根据概率划分落区， 默认50%
    :param x:
    :param thr:
    :return:
    """
    return xr.where(x >= thr, np.short(1), np.short(0))


def save_to_netcdf(nc, savepath, close=True):
    if not os.path.isdir(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    _encode = {"zlib": True, "complevel": 5}
    encoding = dict.fromkeys(list(nc.data_vars), _encode)
    nc.to_netcdf(savepath, format="NetCDF4", encoding=encoding)
    if close:
        nc.close()


def deal_sep(nc_raw,t_, cal_pot=True, cal_pohr=True, cal_poh=True, cal_posw=True,
             prob_dir=None, region='Qinghai'):
    # todo 整理代码
    # todo 稳定性检查
    if prob_dir is None:
        raise AttributeError("没有指定数据存储目录。")
    cals = [cal_pot, cal_pohr, cal_poh, cal_posw]

    min_poh = 2
    min_pot = 2
    min_posw = 2
    min_pohr = 2

    probs_list = []

    thr_zohr = THR_ZOHR
    thr_zoh = THR_ZOH
    thr_zot = THR_ZOT
    thr_zosw = THR_ZOSW

    # sfname = os.path.basename(path)
    # t_ = pd.to_datetime(sfname.split('_')[1])
    st = t_ - timedelta(hours=2)
    try:
        dynamic_thr = DynamicThreshold(st, t_)
        threshold = dynamic_thr.thrs
        print(threshold)
    except:
        traceback.print_exc()
        threshold = {}

    if True in cals:
        # nc_raw = xr.open_dataset(path, decode_times=False)
        ctt = nc_raw.CTT
        ctp = nc_raw.CTP
        cth = nc_raw.CTH
        qpe = nc_raw.QPE
        vap = nc_raw.Channel09
        ir = nc_raw.Channel12

        for prod in [ctt, ctp, cth, qpe, vap, ir]:
            for i in range(ctt.shape[0]):
                try:
                    edges = filters.sobel(np.isnan(prod.data[i]))
                    d = np.where(edges > 0, np.nan, prod.data[i])
                    p = np.percentile(d[~np.isnan(d)], [25, 75])
                    iqr = p[1] - p[0]
                    up_limit = p[1] + iqr * 1.5
                    down_limit = p[0] - iqr * 1.5
                    d_ = np.where((d > up_limit) | (d < down_limit), np.nan, d)
                    d_ = ndimage.median_filter(d_, size=7)
                    prod.data[i] = d_

                except Exception as e:
                    continue

    if cal_pot:
        # 默认参数
        ctt_a, ctt_b = -0.097, 255
        ctp_a, ctp_b = -0.012, 350
        cth_a, cth_b = -5.82E-4, 8200
        qpe_a, qpe_b = -0.6, 4
        vap_a, vap_b = -0.097, 255
        ir_a, ir_b = -0.097, 255
        thrs = threshold.get('POT')
        if thrs is not None:
            thr_ctt, thr_ctp, thr_cth, thr_qpe, thr_vap, thr_ir = thrs
            if thr_ctt is not None:
                ctt_b = thr_ctt[-1]
            if thr_ctp is not None:
                ctp_b = thr_ctp[-1]
            if thr_cth is not None:
                cth_b = thr_cth[0]
            if thr_qpe is not None:
                qpe_b = thr_qpe[0]
            if thr_vap is not None:
                vap_b = thr_vap[-1]
            if thr_ir is not None:
                ir_b = thr_ir[-1]

        pot_ctt = sigmoid(ctt, a=ctt_a, b=ctt_b, reverse=True)
        pot_cth = sigmoid(cth, a=cth_a, b=cth_b)
        pot_ctp = sigmoid(ctp, a=ctp_a, b=ctp_b, reverse=True)
        pot_qpe = sigmoid(qpe, a=qpe_a, b=qpe_b)
        pot_vap = sigmoid(vap, a=vap_a, b=vap_b, reverse=True)
        pot_ir = sigmoid(ir, a=ir_a, b=ir_b, reverse=True)
        wt = 3
        wh = 1
        wp = 2
        wqpe = 1
        wvap = 1
        wir = 1

        # pot = (wt * pot_ctt + wh * pot_cth + wp * pot_ctp) / (wt + wh + wp)
        pot = pot_ctt.copy()
        pot.data = np.nanmean([pot_ctt] * wt + [pot_ctp] * wp + [pot_cth] * wh +
                              [pot_qpe] * wqpe + [pot_vap] * wvap + [pot_ir] * wir,
                              axis=0)

        pot.data = np.where(pot.data >= min_pot, pot.data, 0)
        pot.name = 'POT'
        pot = pot.to_dataset().astype(np.short)
        pot.POT.attrs = dict(units='%', scale_factor=1,
                             add_offset=0, _FillValue=0)

        zot = zone_identify(pot.POT, thr=thr_zot).astype(np.short)
        for i in range(zot.shape[0]):
            selem = np.ones((3, 3))
            zot.data[i] = morphology.binary_opening(zot.data[i], selem=selem)
            zot.data[i] = ndimage.median_filter(zot.data[i], size=7)
            zot.data[i] = ndimage.median_filter(zot.data[i], size=3)
            mask = morphology.remove_small_objects(zot.data[i] > 0,
                                                   min_size=50)
            anom = zot.data[i] - mask
            zot.data[i] = zot.data[i] * mask

            c_ = xr.where(anom == 0, pot.POT[i], np.nan)
            c = xr.where(c_ * (~mask) >= thr_zot, np.nan, c_)
            d = c.interpolate_na(dim='longitude', method='linear',
                                 limit=7)
            pot.POT.data[i] = d.data

        pot['ZOT'] = zot.astype(np.short)
        pot.ZOT.attrs = dict(units='', scale_factor=1,
                             add_offset=0, _FillValue=0)

        probs_list.append(pot)

    if cal_poh:
        ctt_a, ctt_b = -0.1, 240
        ctp_a, ctp_b = -0.023, 300
        cth_a, cth_b = -1.2E-3, 9500
        qpe_a, qpe_b = -0.6, 4
        vap_a, vap_b = -0.1, 240
        ir_a, ir_b = -0.1, 240
        thrs = threshold.get('POH')
        if thrs is not None:
            thr_ctt, thr_ctp, thr_cth, thr_qpe, thr_vap, thr_ir = thrs
            if thr_ctt is not None:
                ctt_b = thr_ctt[-1]
            if thr_ctp is not None:
                ctp_b = thr_ctp[-1]
            if thr_cth is not None:
                cth_b = thr_cth[0]
            if thr_qpe is not None:
                qpe_b = thr_qpe[0]
            if thr_vap is not None:
                vap_b = thr_vap[-1]
            if thr_ir is not None:
                ir_b = thr_ir[-1]

        poh_ctt = sigmoid(ctt, a=ctt_a, b=ctt_b, reverse=True)
        poh_cth = sigmoid(cth, a=cth_a, b=cth_b)
        poh_ctp = sigmoid(ctp, a=ctp_a, b=ctp_b, reverse=True)
        poh_qpe = sigmoid(qpe, a=qpe_a, b=qpe_b)
        poh_vap = sigmoid(vap, a=vap_a, b=vap_b, reverse=True)
        poh_ir = sigmoid(ir, a=ir_a, b=ir_b, reverse=True)
        wt = 3
        wh = 1
        wp = 2
        wqpe = 1
        wvap = 1
        wir = 1

        # poh = (wt * poh_ctt + wh * poh_cth + wp * poh_ctp) / (wt + wh + wp)
        poh = poh_ctt.copy()
        poh.data = np.nanmean([poh_ctt] * wt + [poh_ctp] * wp + [poh_cth] * wh +
                              [poh_qpe] * wqpe + [poh_vap] * wvap + [poh_ir] * wir,
                              axis=0)
        poh.data = np.where(poh.data >= min_poh, poh.data, 0)
        poh.name = 'POH'
        poh = poh.to_dataset().astype(np.short)
        poh.POH.attrs = dict(units='%', scale_factor=1,
                             add_offset=0, _FillValue=0)
        zoh = zone_identify(poh.POH, thr=thr_zoh).astype(np.short)
        for i in range(zoh.shape[0]):
            selem = np.ones((3, 3))
            zoh.data[i] = morphology.binary_opening(zoh.data[i], selem=selem)
            zoh.data[i] = ndimage.median_filter(zoh.data[i], size=7)
            zoh.data[i] = ndimage.median_filter(zoh.data[i], size=3)
            mask = morphology.remove_small_objects(zoh.data[i] > 0,
                                                   min_size=50)
            anom = zoh.data[i] - mask
            zoh.data[i] = zoh.data[i] * mask

            c_ = xr.where(anom == 0, poh.POH[i], np.nan)
            c = xr.where(c_ * (~mask) >= thr_zoh, np.nan, c_)
            d = c.interpolate_na(dim='longitude', method='linear',
                                 limit=7)
            poh.POH.data[i] = d.data

        poh['ZOH'] = zoh.astype(np.short)
        poh.ZOH.attrs = dict(units='', scale_factor=1,
                             add_offset=0, _FillValue=0)

        probs_list.append(poh)

    if cal_posw:
        ctt_a, ctt_b = -0.08, 255
        ctp_a, ctp_b = -0.012, 350
        cth_a, cth_b = -5.8E-4, 7500
        qpe_a, qpe_b = -0.6, 4
        vap_a, vap_b = -0.08, 255
        ir_a, ir_b = -0.08, 255
        thrs = threshold.get('POSW')
        if thrs is not None:
            thr_ctt, thr_ctp, thr_cth, thr_qpe, thr_vap, thr_ir = thrs
            if thr_ctt is not None:
                ctt_b = thr_ctt[-1]
            if thr_ctp is not None:
                ctp_b = thr_ctp[-1]
            if thr_cth is not None:
                cth_b = thr_cth[0]
            if thr_qpe is not None:
                qpe_b = thr_qpe[0]
            if thr_vap is not None:
                vap_b = thr_vap[-1]
            if thr_ir is not None:
                ir_b = thr_ir[-1]

        posw_ctt = sigmoid(ctt, a=ctt_a, b=ctt_b, reverse=True)
        posw_cth = sigmoid(cth, a=cth_a, b=cth_b)
        posw_ctp = sigmoid(ctp, a=ctp_a, b=ctp_b, reverse=True)
        posw_qpe = sigmoid(qpe, a=qpe_a, b=qpe_b)
        posw_vap = sigmoid(vap, a=vap_a, b=vap_b, reverse=True)
        posw_ir = sigmoid(ir, a=ir_a, b=ir_b, reverse=True)
        wt = 2
        wh = 1
        wp = 2
        wqpe = 1
        wvap = 2
        wir = 2

        # posw = (wt * posw_ctt + wh * posw_cth + wp * posw_ctp) / (wt + wh + wp)
        posw = posw_ctt.copy()
        posw.data = np.nanmean([posw_ctt] * wt + [posw_ctp] * wp + [posw_cth] * wh +
                               [posw_qpe] * wqpe + [posw_vap] * wvap + [posw_ir] * wir,
                               axis=0)
        posw.data = np.where(posw.data >= min_posw, posw.data, 0)
        posw.name = 'POSW'
        posw = posw.to_dataset().astype(np.short)
        posw.POSW.attrs = dict(units='%', scale_factor=1,
                               add_offset=0, _FillValue=0)
        zosw = zone_identify(posw.POSW, thr=thr_zosw).astype(np.short)
        for i in range(zosw.shape[0]):
            selem = np.ones((3, 3))
            zosw.data[i] = morphology.binary_opening(zosw.data[i], selem=selem)
            zosw.data[i] = ndimage.median_filter(zosw.data[i], size=7)
            zosw.data[i] = ndimage.median_filter(zosw.data[i], size=3)
            mask = morphology.remove_small_objects(zosw.data[i] > 0,
                                                   min_size=50)
            anom = zosw.data[i] - mask
            zosw.data[i] = zosw.data[i] * mask

            c_ = xr.where(anom == 0, posw.POSW[i], np.nan)
            c = xr.where(c_ * (~mask) >= thr_zosw, np.nan, c_)
            d = c.interpolate_na(dim='longitude', method='linear',
                                 limit=7)
            posw.POSW.data[i] = d.data

        posw['ZOSW'] = zosw.astype(np.short)
        posw.ZOSW.attrs = dict(units='', scale_factor=1,
                               add_offset=0, _FillValue=0)

        probs_list.append(posw)

    if cal_pohr:
        ctt_a, ctt_b = -0.08, 230
        ctp_a, ctp_b = -0.013, 260
        cth_a, cth_b = -5E-4, 10000
        qpe_a, qpe_b = -0.6, 4
        vap_a, vap_b = -0.08, 230
        ir_a, ir_b = -0.08, 230
        thrs = threshold.get('POHR')
        if thrs is not None:
            thr_ctt, thr_ctp, thr_cth, thr_qpe, thr_vap, thr_ir = thrs
            if thr_ctt is not None:
                ctt_b = thr_ctt[-1]
            if thr_ctp is not None:
                ctp_b = thr_ctp[-1]
            if thr_cth is not None:
                cth_b = thr_cth[0]
            if thr_qpe is not None:
                qpe_b = thr_qpe[0]
            if thr_vap is not None:
                vap_b = thr_vap[-1]
            if thr_ir is not None:
                ir_b = thr_ir[-1]

        pohr_ctt = sigmoid(ctt, a=ctt_a, b=ctt_b, reverse=True)
        pohr_cth = sigmoid(cth, a=cth_a, b=cth_b)
        pohr_ctp = sigmoid(ctp, a=ctp_a, b=ctp_b, reverse=True)
        pohr_qpe = sigmoid(qpe, a=qpe_a, b=qpe_b)
        pohr_vap = sigmoid(vap, a=vap_a, b=vap_b, reverse=True)
        pohr_ir = sigmoid(ir, a=ir_a, b=ir_b, reverse=True)
        wt = 2
        wh = 1
        wp = 1
        wqpe = 3
        wvap = 1
        wir = 1

        # pohr = (wt * pohr_ctt + wh * pohr_cth + wp * pohr_ctp) / (wt + wh + wp)
        pohr = pohr_ctt.copy()
        pohr.data = np.nanmean([pohr_ctt] * wt + [pohr_ctp] * wp + [pohr_cth] * wh +
                               [pohr_qpe] * wqpe + [pohr_vap] * wvap + [pohr_ir] * wir,
                               axis=0)
        pohr.data = np.where(pohr.data >= min_pohr, pohr.data, 0)
        pohr.name = 'POHR'
        pohr = pohr.to_dataset().astype(np.short)
        pohr.POHR.attrs = dict(units='%', scale_factor=1,
                               add_offset=0, _FillValue=0)
        zohr = zone_identify(pohr.POHR, thr=thr_zohr).astype(np.short)
        for i in range(zohr.shape[0]):
            selem = np.ones((3, 3))
            zohr.data[i] = morphology.binary_opening(zohr.data[i], selem=selem)
            zohr.data[i] = ndimage.median_filter(zohr.data[i], size=7)
            zohr.data[i] = ndimage.median_filter(zohr.data[i], size=3)
            mask = morphology.remove_small_objects(zohr.data[i] > 0,
                                                   min_size=50)
            anom = zohr.data[i] - mask
            zohr.data[i] = zohr.data[i] * mask

            c_ = xr.where(anom == 0, pohr.POHR[i], np.nan)
            c = xr.where(c_ * (~mask) >= thr_zohr, np.nan, c_)
            d = c.interpolate_na(dim='longitude', method='linear',
                                 limit=7)
            pohr.POHR.data[i] = d.data

        pohr['ZOHR'] = zohr.astype(np.short)
        pohr.ZOHR.attrs = dict(units='', scale_factor=1,
                               add_offset=0, _FillValue=0)

        probs_list.append(pohr)

    if len(probs_list):
        prob = xr.merge(probs_list)
        # prob = prob.drop_vars(['POHR', 'POH', 'POT', 'POSW'])
        lons = np.arange(89, 104.003, 0.01)
        lats = np.arange(31, 40.003, 0.01)
        prob = prob.interp(dict(longitude=lons,
                                latitude=lats),
                           method="nearest").astype(np.short)
        prob.attrs = nc_raw.attrs
        prob.attrs['title'] = "Probability of Strong Convetive Weather."
        t_str = t_.strftime('%Y%m%d%H%M%S')
        prob_svname = '_'.join([region, 'Nowcasting', t_str, 'All.nc'])
        prob_savepath = os.path.join(prob_dir, region, f'{t_.year:04d}', f'{t_.month:02d}',
                                     f'{t_.day:02}', prob_svname)

        save_to_netcdf(prob, prob_savepath, close=True)

    else:
        print("没有结果生成。")

    if True in cals:
        nc_raw.close()


def get_fpath(t_, sate_dir=None, region='Qinghai'):
    # todo 如果是0点，需要查看前一天的
    # 不超过t_时刻的数据，注意实况和预报的分别
    if sate_dir is None:
        raise AttributeError("没有指定卫星数据目录。")
    pathname = os.path.join(sate_dir, region, f'{t_.year:04d}', f'{t_.month:02d}',
                            f'{t_.day:02}', f'{region}*SEP.nc')
    fpaths = sorted(glob.glob(pathname))
    fpath = None
    for fp in fpaths:
        sfname = os.path.basename(fp)
        ft_ = pd.to_datetime(sfname.split('_')[1])
        if ft_ > t_:
            break
        else:
            fpath = fp

    return fpath


def get_satellite(timespan: List) -> dict:
    """
    根据时间范围获取青海省风云4a数据。

    :param timespan: 起讫时间。
    :return: 数据字典，以时间为键。
    """
    # 根据需要设置
    datas = defaultdict(dict)
    source = 'FY4A'
    ptypes = ['CTT', 'CTP', 'CTH', 'QPE', 'FDI']
    paths = []
    for ptype in ptypes:
        paths_ = fileManager.get_path(source, timespan=timespan, instrument='AGRI',
                                      zone=None, level=None, ptype=ptype,
                                      resolution=None, mode=None, channel=None,
                                      proj=None, suffix='.NC')
        paths.extend(paths_)
    for path in paths:
        try:
            data_t, data_obj = fileReader.get_fy4a_by_path(path)
            data = data_obj.get('obj')
            attrs = data_obj.get('attrs')
            data.load()
            if attrs[1] == 'FDI':
                drop_vars = [var for var in list(data.data_vars) if var != 'Channel09']
                c09 = data.drop_vars(drop_vars)
                drop_vars = [var for var in list(data.data_vars) if var != 'Channel12']
                c12 = data.drop_vars(drop_vars)
                datas['Channel09'].update({data_t: c09})
                datas['Channel12'].update({data_t: c12})
            elif attrs[1] == 'QPE':
                data = data.rename({'Precipitation': 'QPE'})
                datas[attrs[1]].update({data_t: data})

            else:
                datas[attrs[1]].update({data_t: data})
        except:
            continue

    return datas



def run(t_):
    # 预报
    # 获取未来最近的预报时间fcst（在get_fpath中找不超过fcst的预报)
    # 往后一个时次
    cal_fcst = True
    if cal_fcst:
        fcst = t_ + timedelta(minutes=9)
        fcst = fcst.replace(second=59, microsecond=999999)
        fpath_sep = get_fpath(fcst, sate_dir=SEP_DIR, region=REGION)
        if fpath_sep is not None:
            try:
                deal_sep(fpath_sep, cal_pot=True, cal_pohr=True,
                         cal_poh=True, cal_posw=True, prob_dir=PROB_DIR,
                         region=REGION)

            except OSError:
                import time
                time.sleep(10)
                try:
                    deal_sep(fpath_sep, cal_pot=True, cal_pohr=True,
                             cal_poh=True, cal_posw=True, prob_dir=PROB_DIR,
                             region=REGION)
                except:
                    traceback.print_exc()

            except:
                traceback.print_exc()

        else:
            print('没找到预报文件路径。')
            pass


def get_satellite(timespan: List) -> dict:
    """
    根据时间范围获取青海省风云4a数据。

    :param timespan: 起讫时间。
    :return: 数据字典，以时间为键。
    """
    # 根据需要设置
    datas = defaultdict(dict)
    source = 'FY4A'
    ptypes = ['CTT', 'CTP', 'CTH', 'QPE', 'FDI']
    paths = []
    for ptype in ptypes:
        paths_ = fileManager.get_path(source, timespan=timespan, instrument='AGRI',
                                      zone=None, level=None, ptype=ptype,
                                      resolution=None, mode=None, channel=None,
                                      proj=None, suffix='.NC')
        paths.extend(paths_)
    for path in paths:
        try:
            data_t, data_obj = fileReader.get_fy4a_by_path(path)
            data = data_obj.get('obj')
            attrs = data_obj.get('attrs')
            data.load()
            if attrs[1] == 'FDI':
                drop_vars = [var for var in list(data.data_vars) if var != 'Channel09']
                c09 = data.drop_vars(drop_vars)
                drop_vars = [var for var in list(data.data_vars) if var != 'Channel12']
                c12 = data.drop_vars(drop_vars)
                datas['Channel09'].update({data_t: c09})
                datas['Channel12'].update({data_t: c12})
            elif attrs[1] == 'QPE':
                data = data.rename({'Precipitation': 'QPE'})
                datas[attrs[1]].update({data_t: data})

            else:
                datas[attrs[1]].update({data_t: data})
        except:
            continue

    return datas

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--stime', dest='startt',
                        type=str, default=str(datetime.utcnow()),
                        help='开始时间(UTC时间)，格式YYYYMMDDhhmmss，精确到分钟或秒，'
                             '默认系统当前UTC时间。如: --time=202010011200')

    return parser


def pool_run():



if __name__ == "__main__":
    args = build_parser().parse_args()
    st = pd.to_datetime(args.startt)

    try:
        run(st)

    except KeyboardInterrupt:
        raise KeyboardInterrupt(u"终止.")

    except:
        traceback.print_exc()
