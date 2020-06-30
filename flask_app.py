from flask import Flask, request, render_template, redirect, jsonify, send_from_directory
from meme_utils.meme_captioner import MemeCaptioner
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
import base64
import gc

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

mc = MemeCaptioner()
app = Flask(__name__)

@app.route('/', methods=['GET'])
def send_index():
    gc.collect()
    return send_from_directory('./www', "index.html")

@app.route('/<path:path>', methods=['GET'])
def send_root(path):

	gc.collect()
	return send_from_directory('./www', path)

@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if 'image' not in request.files:
        gc.collect()
        return jsonify({'error':'No posted image. Should be attribute named image.'})
    
    file = request.files['image'] 

    if file.filename == '':
        gc.collect()
        return jsonify({'error':'Empty filename submitted.'})
    
    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)
        
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        
        byte_file = file.read()
        
        img = Image.open(BytesIO(byte_file)).convert('RGB')
        
        img, caption = mc.generate_caption(byte_file, byte_file=True)
	
        byte_file = BytesIO()
        img.save(byte_file, format='JPEG')
        byte_file = base64.b64encode(byte_file.getvalue())
        
        items = {'caption':caption,'img':str(byte_file)[2:-1]}

        response = {'pred':items}
        
        del byte_file
        
        gc.collect()
              
        return jsonify(response)
        
    else:
    	gc.collect()

    	return jsonify({'error':'File has invalid extension'})
    
if __name__ == '__main__':
    app.run(debug=False)