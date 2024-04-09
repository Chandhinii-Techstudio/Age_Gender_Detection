from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from age_inf import age_inference
from gen_inf import gen_inference
import base64
HOST = '0.0.0.0' # Standard loopback interface address (localhost)
PORT = 3000  # Port to listen on (non-privileged ports are > 1023)
app = Flask(__name__)
app.config.from_object(__name__)
CORS(app)
@app.route('/userDetails', methods=['POST'])
def result():
        image_64_decode = base64.b64decode(request.json["encodedImage"])
        image_result = open('./Kinetosis/InfImage/mock.jpg', 'wb') # create a writable image and write the decoding result
        image_result.write(image_64_decode)
        age = age_inference()
        gender = gen_inference()
        return { 
        'age': str(age),
        'gender': str(gender)
        }
if __name__ == '__main__':
     app.run(HOST, port=PORT)