import os
import sys
import random
from flask import Flask, Response, request, jsonify ,render_template, redirect, url_for, send_from_directory,send_file
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import time
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

app.app_context().push()


class TrackedObjects(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ship_image = db.Column(db.String(100), nullable=False)
    last_known_location = db.Column(db.String(100), nullable=False)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/config")
def config():
    with open("configuration.json", "r") as config_reader:
        config = eval(config_reader.read())
    # getting the number of events in the database
    events = TrackedObjects.query.all()
    config["events"] = len(events)
    return jsonify(config)

    
@app.route("/mode/<mode>")
def set_mode(mode):
    with open("configuration.json", "r") as config_reader:
        config = eval(config_reader.read())
    modes = ['auto', 'man']
    if mode in modes:
        if mode == 'auto':
            config['manauto'] = True
        if mode == "man":
            config["manauto"] = False
        with open("configuration.json", "w") as config_writer:
            config_writer.write(str(config)) 
        return redirect(url_for("index"))
    else:
        return "Invalid Mode: It Should be either /mode/auto or /mode/man for Operations"

def generate_img():
    last_frame = None
    while True:
        try:
            time.sleep(0.1)
            image = cv2.imread("frame.jpg")
            _, frame = cv2.imencode(".jpg", image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        except:
            continue

@app.route("/mjpeg")
def mjpeg():
    return Response(generate_img(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/tempdb")
def tempdb():
    with open("temp.json", "r") as temp_db_reader:
        temp_db = eval(temp_db_reader.read())
    return jsonify(temp_db)
@app.route("/errorlog")
def errorlog():
    with open("errors.json") as error_log:
        log = eval(error_log.read())
    return jsonify(log)
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, threaded=True, port=5000)