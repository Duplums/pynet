import visdom
import numpy as np
from subprocess import Popen, PIPE
import sys

class Visualizer:

    def __init__(self, history, port='8097', env='main', server="http://localhost"):
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
        self.display_ids = {metric: id for id, metric in enumerate(self.history.metrics)}
        self.free_display_id = len(self.history.metrics)

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
        all_steps = self.history.steps
        self.refresh_display_ids_metrics()
        if len(all_steps) > 0 and type(all_steps[0]) == tuple:
            # We assume that the steps are formated as (epoch, iteration)
            max_iter = max([it for (epoch, it) in all_steps])
            x_axis = [epoch + float(it)/(max_iter+1) for (epoch, it) in all_steps]
        else:
            x_axis = all_steps
        for (metric, id) in self.display_ids.items():
            self.vis.line(X=x_axis,
                          Y=self.history[metric][1],
                          opts={'title': metric + ' per epoch',
                                'xlabel': 'epoch',
                                'ylabel': metric},
                          win=id)

    def display_images(self, images, labels, ncols=1):
        """
        :param images: numpy array with shape NxCxHxW(xD) (N=nb of images, C=nb of channels H=height, W=width, D=depth)
        :param labels: list of N label (str)
        :param ncols: int representing how many pictures we're putting on the same row
        """
        assert len(images.shape) in [4, 5], \
            "The images must have a shape NxCxHxW or NxCxHxWxD, unsupported shape {}".format(images.shape)
        assert len(images) == len(labels), "Input images ({}) must all have labels ({})".format(len(images), len(labels))
        (N,C,H,W) = images.shape[:4]
        D = images.shape[4] if len(images.shape) == 5 else None

        # First, rescale the images
        images = np.nan_to_num([(img - np.min(img))/(np.max(img) - np.min(img)) for img in images])

        # If the images are 3D, display only the 3 cuts: x-axis, y-axis and z-axis
        if D is not None:
            ncols = 3 * ncols
            labels = np.array([['%s (x-axis)'% l, '%s (y-axis)'% l, '%s (z-axis)'% l] for l in labels]).flatten()
            list_images = []
            for img in images:
                list_images.extend([img[:, H//2, :, :], img[:, :, W//2, :], img[:, :, :, D//2]])
            images = list_images
            N = 3*N

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

        self.vis.images(images, nrow=ncols, win=self.free_display_id,
                        padding=2)
        #self.vis.text(table_css + label_html, win=self.free_display_id + 1)


