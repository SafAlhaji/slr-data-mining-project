from flask import Flask
from flask_socketio import SocketIO

class SocketConnection:
    HOST = 'http://127.0.0.1'
    PORT = 65432
    main = None

    def __init__(self):
        self.app = Flask(__name__)
        sio = SocketIO(self.app, logger=True, engineio_logger=True, cors_allowed_origins="*")
        self.server = sio

        @sio.on('connect')
        def connect(auth):
            self.server.start_background_task(target=self.main.__main__)

        @sio.on('predict')
        def predict(data):
            requestedPrice = float(data['area'])
            self.main.setRequestedPrice(requestedPrice)
            prediction = self.main.regression.predict([requestedPrice])
            self.server.emit('prediction', { 'price': prediction.tolist()[0][0] })

    def set_main(self, main):
        self.main = main

    def open(self):
        self.server.run(self.app, host='127.0.0.1')
        
