import xgboost as xgb
import numpy as np


def map_for_dict_Gender(Gender):
    dict_Gender = {'Male': 0, 'Female': 1}
    res = dict_Gender.get(Gender)
    return res


def map_for_dict_MariStat(MariStat):
    dict_MariStat = {'Other': 0, 'Alone': 1}
    res = dict_MariStat.get(MariStat)
    return res


def f_VehUsage(VehUsage, value):
    if VehUsage == value:
        VehUsage_dummy = 1
    else:
        VehUsage_dummy = 0
    return VehUsage_dummy


def f_SocioCateg(SocioCateg, value):
    if SocioCateg == value:
        SocioCateg_dummy = 1
    else:
        SocioCateg_dummy = 0
    return SocioCateg_dummy


def process_input(json_input):
    Exposure = json_input["Exposure"]
    LicAge = json_input["LicAge"]
    Gender = map_for_dict_Gender(json_input["Gender"])
    MariStat = map_for_dict_MariStat(json_input["MariStat"])
    DrivAge = json_input["DrivAge"]
    HasKmLimit = json_input["HasKmLimit"]
    BonusMalus = json_input["BonusMalus"]
    OutUseNb = json_input["OutUseNb"]
    RiskArea = json_input["RiskArea"]

    VehUsg_Private = f_VehUsage(json_input["VehUsage"], 'Professional')
    VehUsg_Private_trip_to_office = f_VehUsage(json_input["VehUsage"], 'Private+trip to office')
    VehUsg_Professional = f_VehUsage(json_input["VehUsage"], 'Professional')
    VehUsg_Professional_run = f_VehUsage(json_input["VehUsage"], 'Professional run')

    SocioCateg_CSP1 = f_SocioCateg(json_input["SocioCateg"], "CSP1")
    SocioCateg_CSP2 = f_SocioCateg(json_input["SocioCateg"], "CSP2")
    SocioCateg_CSP3 = f_SocioCateg(json_input["SocioCateg"], "CSP3")
    SocioCateg_CSP4 = f_SocioCateg(json_input["SocioCateg"], "CSP4")
    SocioCateg_CSP5 = f_SocioCateg(json_input["SocioCateg"], "CSP5")
    SocioCateg_CSP6 = f_SocioCateg(json_input["SocioCateg"], "CSP6")
    SocioCateg_CSP7 = f_SocioCateg(json_input["SocioCateg"], "CSP7")

    dmatrix = xgb.DMatrix(np.array([[
        Exposure,
        LicAge,
        Gender,
        MariStat,
        DrivAge,
        HasKmLimit,
        BonusMalus,
        OutUseNb,
        RiskArea,
        VehUsg_Private,
        VehUsg_Private_trip_to_office,
        VehUsg_Professional,
        VehUsg_Professional_run,
        SocioCateg_CSP1,
        SocioCateg_CSP2,
        SocioCateg_CSP3,
        SocioCateg_CSP4,
        SocioCateg_CSP5,
        SocioCateg_CSP6,
        SocioCateg_CSP7,
        DrivAge ** 2]]))

    return dmatrix
