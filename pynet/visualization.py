import visdom
import numpy as np
import torch
from subprocess import Popen, PIPE
import sys

class Visualizer:

    def __init__(self, history, port='8097', env='main', server="http://localhost", offset_win=0):
        """ This class aims to gather plotting tools interesting during a training or a test. Basically, it uses the
        history obj created during a training to plot all the metrics. It can also display images (for instance during
        the training of the autoencoder)
        :param  history: a History object that contains all the metrics + objective function computed during a training
                port: (str or int) the port on which the visdom server is launched
                env: (str) the environment to use
                server: (str) the server on which visdom is launched
        """
        self.history = history
        self.port = int(port)
        self.env = env
        self.server = server
        self.vis = visdom.Visdom(port=self.port, env=env, server=server)
        if not self.vis.check_connection():
            self.create_visdom_connections()
        # Init every panel's id for the visualization of each metric
        self.display_ids = {metric: id+offset_win for id, metric in enumerate(self.history.metrics)}
        self.free_display_id = len(self.history.metrics)+offset_win

    def create_visdom_connections(self):
        """It starts a new server on port <self.port>"""
        current_python = sys.executable
        cmd = current_python + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def refresh_display_ids_metrics(self):
        for metric in self.history.metrics:
            if metric not in self.display_ids:
                self.display_ids[metric] = self.free_display_id
                self.free_display_id = self.free_display_id + 1

    def refresh_current_metrics(self):
        self.refresh_display_ids_metrics()
        for (metric, id) in self.display_ids.items():
            x_axis, y = self.history[metric]
            if len(x_axis) > 0 and type(x_axis[0]) == tuple:
                x_axis = np.arange(len(x_axis))
            self.vis.line(X=x_axis,
                          Y=self.history[metric][1],
                          opts={'title': metric + ' per iterations',
                                'xlabel': 'iterations',
                                'ylabel': metric},
                          win=id)


    def visualize_data(self, inputs, outputs, num_samples=None):
        from pynet.utils import tensor2im
        assert inputs.shape == outputs.shape
        # in our case, inputs == targets

        valid_indices = [idx for idx in range(len(inputs)) if inputs[idx].max() > 0]

        if num_samples:
            valid_indices = np.random.choice(valid_indices, num_samples, replace=False)

        inputs = inputs[valid_indices]
        outputs = outputs[valid_indices]

        N, b_shape = inputs.shape[0], inputs.shape[1:]
        visuals = np.array([tensor2im(inputs), tensor2im(outputs)], dtype=np.float32) # shape 2xNxCxHxW(xD) (if 3D)
        visuals = visuals.swapaxes(0, 1).reshape((2*N,) + b_shape)
        labels = np.array([['Input pic {}'.format(k), 'Output pic {}'.format(k)] for k in range(inputs.shape[0])]).flatten()

        self.display_images(visuals, labels, ncols=2)


    def t_SNE(self, features, labels):

        self.vis.embeddings(features, labels)

    def display_images(self, images, labels=None, ncols=None, middle_slices=False):
        """
        :param images: numpy array with shape NxCxHxW(xD) (N=nb of images, C=nb of channels H=height, W=width, D=depth)
        :param labels: list of N label (str)
        :param ncols: int representing how many pictures we're putting on the same row
        :param middle_slices: bool, whether to cut the image in the middle or not
        """
        if images is None:
            return

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        assert len(images.shape) in [4, 5], \
            "The images must have a shape NxCxHxW or NxCxHxWxD, unsupported shape {}".format(images.shape)
        if labels is not None:
            assert len(images) == len(labels), "Input images ({}) must all have labels ({})".\
                format(len(images), len(labels))
        else:
            labels = ["Image %i" % i for i in range(len(images))]

        (N,C,H,W) = images.shape[:4]
        D = images.shape[4] if len(images.shape) == 5 else None
        ncols = ncols or (1 if D is None else 3)
        # First, rescale the images
        images = np.nan_to_num([(img - np.min(img))/(np.max(img) - np.min(img)) for img in images])

        table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (W, H)  # create a table css

        # Fill the HTML table with the labels
        label_html, label_html_row = '', ''
        for n in range(N):
            label_html_row += '<td>%s</td>' % labels[n]
            if n % ncols == 0 and n > 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row

        # Display the images in one visdom panel
        if D is not None:
            slice_H = H//2 if middle_slices else np.argmax(np.sum(np.abs(images), axis=(0,1,3,4)))
            self.vis.images(images[:,:,slice_H,:,:], nrow=ncols, win=self.free_display_id,
                            padding=2, opts=dict(title="X-axis cut accross subjects"))
            slice_W = W//2 if middle_slices else np.argmax(np.sum(np.abs(images), axis=(0,1,2,4)))
            self.vis.images(images[:,:,:,slice_W,:], nrow=ncols, win=self.free_display_id+1,
                            padding=2, opts=dict(title="Y-axis cut accross subjects"))
            slice_D = D//2 if middle_slices else np.argmax(np.sum(np.abs(images), axis=(0,1,2,3)))
            self.vis.images(images[:,:,:,:,slice_D], nrow=ncols, win=self.free_display_id+2,
                            padding=2, opts=dict(title="Z-axis cut accross subjects"))
        else:
            self.vis.images(images, nrow=ncols, win=self.free_display_id,
                            padding=2, opts=dict(title="All images"))

        #self.vis.text(table_css + label_html, win=self.free_display_id + 1)


