# from crypt import methods
from flask import Flask, render_template, request, Response, redirect, url_for
import pandas as pd

from detection import *

app = Flask(__name__)

VIDEO = VideoStreaming()
COUNTER = ObjectDetection()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO.show(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/result')
def result():
    hasil_counter = COUNTER.show_counter()
    data = COUNTER.show_chart()
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    print("ini labels" + str(labels))
    return render_template('result.html', hc = hasil_counter, labels=labels, values=values )

@app.route('/request_preview_switch')
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print('*'*10, VIDEO.preview)
    return "nothing"

@app.route('/request_model_switch')
def request_model_switch():
    VIDEO.detect = not VIDEO.detect
    print('*'*10, VIDEO.detect)
    return "nothing"



# @app.route('/result_counter', methods=['POST'])
# def result_counter():
#     counter_hasil = int(request.form[COUNTER.counter])
#     return str(counter_hasil)

# @app.route('/result_chart')
# def result_chart():
#     return response(
#         COUNTER.show_chart()
#     )


if __name__ == '__main__':
    app.run(debug=True)