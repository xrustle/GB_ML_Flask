import requests

data = {
    "ID": 1,
    "Exposure": 0.249,
    "LicAge": 453,
    "Gender": "Male",
    "MariStat": "Other",
    "DrivAge": 59,
    "HasKmLimit": 0,
    "BonusMalus": 50,
    "OutUseNb": 2,
    "RiskArea": 2,
    "VehUsage": "Private+trip to office",
    "SocioCateg": "CSP5"
}


def send_json(data):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response


if __name__ == '__main__':
    response = send_json(data)
    print(response.json())
