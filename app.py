import xgboost as xgb
import pickle

from flask import Flask, render_template, redirect, url_for, request, jsonify, send_from_directory
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired

from postman import data, send_json

import json
import os

from process_data import process_input

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

xgb_model = pickle.load(open('xgb_model.pickle', 'rb'))

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class ClientDataForm(FlaskForm):
    id = IntegerField('ID', validators=[DataRequired()])
    exposure = IntegerField('Exposure', validators=[DataRequired()])
    licage = IntegerField('Водительский стаж', validators=[DataRequired()])
    gender = SelectField('Пол', choices=[('Male', 'М'), ('Female', 'Ж')])
    maristat = SelectField('Семейное положение', choices=[('Alone', 'Alone'), ('Other', 'Other')])
    drivage = IntegerField('Возраст', validators=[DataRequired()])
    haskmlimit = SelectField('Ограничение пробега', choices=[('1', 'Есть'), ('0', 'Нет')])
    bonusmalus = IntegerField('Коэффициент BonusMalus', validators=[DataRequired()])
    outuse = IntegerField('Out-of-use за 4 года', validators=[DataRequired()])
    riskarea = SelectField('Risk Area', choices=[(str(i), str(i)) for i in range(1, 14)])
    vehusage = SelectField('Тип использования', choices=[('Private', 'Личный'),
                                                         ('Private + trip to office', 'Личный + поездки на работу'),
                                                         ('Professional', 'Профессиональный водитель'),
                                                         ('Professional run', 'Профессиональные соревнования')])
    sociocateg = SelectField('Социальная категория', choices=[('CSP' + str(i), str(i)) for i in range(1, 8)])


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    json_input = request.json

    # Request logging
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()

    id = json_input['ID']

    dmatrix = process_input(json_input)

    value_xgb = xgb_model.predict(dmatrix)[0]

    result = {
        'ID': id,
        'ClaimsCount': int(value_xgb)
    }

    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datatime} predicted for {duration} msec: {result}\n')

    return jsonify(result)


@app.route("/predict_form", methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    if request.method == 'POST':
        data['ID'] = request.form.get('id')
        data['Exposure'] = float(request.form.get('exposure'))
        data['LicAge'] = float(request.form.get('licage'))
        data['Gender'] = request.form.get('gender')
        data['MariStat'] = request.form.get('maristat')
        data['DrivAge'] = float(request.form.get('drivage'))
        data['HasKmLimit'] = request.form.get('haskmlimit')
        data['BonusMalus'] = float(request.form.get('bonusmalus'))
        data['OutUseNb'] = float(request.form.get('outuse'))
        data['RiskArea'] = float(request.form.get('riskarea'))
        data['VehUsage'] = request.form.get('vehusage')
        data['SocioCateg'] = request.form.get('sociocateg')
        try:
            response = send_json(data)
            response = response.text
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


@app.route("/predicted/<response>")
def predicted(response):
    response = json.loads(response)
    print("@app.route('/predicted/<response>')", response)
    return render_template('predicted.html', response=response)


@app.route("/favicon.ico", methods=['GET'])
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'images'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.errorhandler(Exception)
def exceptions(e):
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    error_message = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 current_datatime,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_message)
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
