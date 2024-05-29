from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, redirect, request, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, FileField
from wtforms.validators import data_required
from werkzeug.utils import secure_filename
import os
import glob


def process_image(img):
    """
    This function processes an image and returns 5 major colors in the image alongside the rgb codes
    :param img:
    :return: A list containing the major colors and the codes
    """
    # Load the image using Pillow
    img_path = img
    image = Image.open(img_path)

    # Convert the image to an RGB format
    image = image.convert('RGB')

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Reshape the image array to a 2D array where each row is a pixel
    pixels = image_np.reshape(-1, 3)

    # Number of clusters (dominant colors) you want to find
    num_colors = 5

    # Use KMeans clustering to find the dominant colors in the image
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the cluster centers (the dominant colors)
    palette = kmeans.cluster_centers_

    # Convert the palette colors from float to integer values
    palette = palette.astype(int)

    # Display the color palette using matplotlib
    plt.figure(figsize=(8, 2))
    plt.imshow([palette], aspect='auto')
    plt.axis('off')
    # plt.show()
    # Save the matplotlib figure to a folder
    plt.savefig('static/processed_image/fig.png')
    processed_image_path = 'static/processed_image/fig.png'
    image_codes = palette
    # Returns a list of the image path and the color codes
    return [processed_image_path, image_codes]


def allowed_file(filename):
    """
    This function checks input files and determines which is allowed and which is not
    :param filename:
    :return: a Boolean
    """
    allowed_ext = filename.split('.')
    allowed_ext = allowed_ext[-1]
    if allowed_ext.lower() in ALLOWED_EXTENSIONS:
        return True
    else:
        return False


ALLOWED_EXTENSIONS = ['blp', 'bmp', 'dib', 'bufr', 'cur', 'pcx', 'dcx', 'dds', 'ps', 'eps', 'fit', 'fits', 'fli', 'flc', 'ftc', 'ftu', 'gbr', 'gif', 'grib', 'h5', 'hdf', 'png', 'apng', 'jp2', 'j2k', 'jpc', 'jpf', 'jpx', 'j2c', 'icns', 'ico', 'im', 'iim', 'jfif', 'jpe', 'jpg', 'jpeg', 'mpg', 'mpeg', 'tif', 'tiff', 'mpo', 'msp', 'palm', 'pcd', 'pdf', 'pxr', 'pbm', 'pgm', 'ppm', 'pnm', 'pfm', 'psd', 'qoi', 'bw', 'rgb', 'rgba', 'sgi', 'ras', 'tga', 'icb', 'vda', 'vst', 'webp', 'wmf', 'emf', 'xbm', 'xpm']


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', '')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

image_path = 'static/uploads/'
jinja_image_path = 'uploads/'


class UploadForm(FlaskForm):
    file = FileField(label="Choose image", validators=[data_required()], render_kw={"class": "form-control"})
    submit = SubmitField(label="Upload", render_kw={"class": "btn btn-outline-success size"})


@app.route('/', methods=["GET", "POST"])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        # A list of the files in the uploaded image folder
        last_uploaded_image = glob.glob(os.path.join('static/uploads/', '*'))
        if last_uploaded_image:
            # Deletes the first item of the list since we are sure is only one item in the list
            os.remove(last_uploaded_image[0])

        # Gets the image uploaded from the clientside of the website and stores it in the variable file
        file = form.file.data
        if file and allowed_file(file.filename):
            # Used secure_filename function from werkzeug package to secure the file and avoid attacks
            original_filename = secure_filename(file.filename)

            # # This line uses the rsplit method to get the file extension but I didn't later use in my program
            # file_extension = original_filename.rsplit('.', 1)[1]
            # new_filename = f'uploaded_image.{file_extension}'
            # Gave the file a new name, so I can easily serve it back to the client side and display it
            new_filename = 'uploaded_image.png'

            # The path to save the file
            file_path = os.path.join('static/uploads/', new_filename)
            # saves the file
            file.save(file_path)

            # passes the file to the process_image function
            processed = process_image(file)
            processed_img = processed[0]
            processed_codes = processed[1]
            return redirect(url_for('index', success=True, processed_img=processed_img, processed_codes=processed_codes))
        else:
            flash("This file format is not allowed")
            return redirect(url_for('index'))
    return render_template("index.html", form=form, success=request.args.get('success'), processed_img=request.args.get('processed_img'), processed_codes=request.args.get('processed_codes'))


if __name__ == "__main__":
    app.run(host='0.0.0.0')

