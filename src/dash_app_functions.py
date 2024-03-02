import dash
from dash import dcc, html
from dash.dependencies import Input, Output
# import plotly.graph_objs as go
import base64
from PIL import Image
# import os
import pickle as pkl
import io


def load_plotly_figure(path, new_x_axes=None):
    with open(path, 'rb') as file:
        figure = pkl.load(file)
    
    if new_x_axes is not None:
        figure.update_xaxes(range = [-new_x_axes[0],new_x_axes[1]])

    # Convert the Plotly figure to a Dash graph object
    graph = dcc.Graph(
        figure=figure
    )

    return graph


def load_static_image(path, image_width:int):
    # Read the image file
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize the image while preserving aspect ratio
        img.thumbnail((image_width, image_width))
        # Convert the image to RGBA if it's not already in that mode
        img = img.convert('RGBA')
        # Create a white background image to place the resized image on
        bg = Image.new('RGBA', (image_width, image_width), (255, 255, 255, 255))
        bg.paste(img, (int((image_width - img.width) / 2), int((image_width - img.height) / 2)), img)
        # Save the resized and centered image to a temporary file
        with io.BytesIO() as temp_image_buffer:
            bg.save(temp_image_buffer, format='PNG')
            temp_image_buffer.seek(0)
            # Encode the resized image to base64 format
            encoded_image = base64.b64encode(temp_image_buffer.read()).decode('utf-8')

    return 'data:image/png;base64,{}'.format(encoded_image)