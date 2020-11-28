import argparse
import os.path
import random
import sys

import numpy as np
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QDialog
)
from skimage import io

from model import FaceVerificator

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
BASE_COLOR = QColor('yellow')
TEXT_COLOR = QColor('yellow')
BASE_WIDTH = 2
BOX_SIZE = 227
TEXT_WIDTH = 2
TEXT_SIZE = 16
TEXT_FONT = QFont('Sans', TEXT_SIZE)
MATCH_BACK_COLOR = QColor('cyan')


class TablePopup(QDialog):
    def __init__(self, scores, comp):
        super().__init__()
        layout = QVBoxLayout(self)
        rows, cols = scores.shape
        table = QTableWidget(rows, cols)
        layout.addWidget(table)

        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem('{:.3f}'.format(scores[i, j]))
                if comp[i, j]:
                    item.setBackground(MATCH_BACK_COLOR)
                table.setItem(i, j, item)

        hh = list(map(lambda s: '2: {}'.format(s), range(cols)))
        vh = list(map(lambda s: '1: {}'.format(s), range(rows)))

        table.setHorizontalHeaderLabels(hh)
        table.setVerticalHeaderLabels(vh)

        self.setLayout(layout)
        self.setWindowTitle('Scores table')
        self.show()
        self.setAttribute(Qt.WA_DeleteOnClose)


def show_error(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle('Error')
    msg.exec_()
    return


def random_color():
    return QColor(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


class Main(QWidget):
    def __init__(self, model_path):
        super().__init__()

        self.image_loaded = [False, False]
        self.image_path = [None, None]
        self.image_data = [None, None]
        self.image_scale = [1.0, 1.0]
        self.image_pixmap = [None, None]

        self.dist = 0.85

        self.matched = False

        self.fv = FaceVerificator(model_path)

        self.setGeometry(300, 300, 1280, 720)
        self.setWindowTitle('Face identification using CNN + TPE')

        self.load1_button = QPushButton('Load 1')
        self.load2_button = QPushButton('Load 2')

        self.match_button = QPushButton('Match')
        self.exit_button = QPushButton('Exit')

        self.image_label = [QLabel(), QLabel()]

        self.exit_button.clicked.connect(QCoreApplication.instance().quit)
        self.load1_button.clicked.connect(self.load1_clicked)
        self.load2_button.clicked.connect(self.load2_clicked)
        self.match_button.clicked.connect(self.match_clicked)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.load1_button)
        hbox1.addWidget(self.load2_button)
        hbox1.addStretch(1)
        hbox1.addWidget(self.match_button)
        hbox1.addStretch(1)
        hbox1.addWidget(self.exit_button)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.image_label[0])
        hbox2.addStretch(1)
        hbox2.addWidget(self.image_label[1])

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        vbox.addLayout(hbox1)

        self.setLayout(vbox)

        self.show()

    def image_file_dialog(self):
        return QFileDialog().getOpenFileName(
            self,
            'Single File',
            './',
            'Image files (*.jpg *.jpeg *.png)'
        )[0]

    def load_image(self, path, n):
        pixmap = QPixmap(path)

        if pixmap.width() > pixmap.height():
            scale = IMAGE_WIDTH / pixmap.width()
        else:
            scale = IMAGE_HEIGHT / pixmap.height()

        if scale < 1.0:
            new_width = int(pixmap.width() * scale)
            new_height = int(pixmap.height() * scale)
            pixmap = pixmap.scaled(new_width, new_height)

        self.image_scale[n] = scale
        self.image_pixmap[n] = pixmap
        self.image_label[n].setPixmap(pixmap)
        self.image_label[n].show()

    def load_clicked(self, n):
        path = self.image_file_dialog()

        if (path is None) or (not os.path.exists(path)):
            return

        self.image_path[n] = path
        self.image_loaded[n] = True
        self.image_data[n] = io.imread(path)
        self.load_image(path, n)

        if n == 0 and self.image_loaded[1]:
            self.load_image(self.image_path[1], 1)
        elif n == 1 and self.image_loaded[0]:
            self.load_image(self.image_path[0], 0)

        self.matched = False

    def load1_clicked(self):
        self.load_clicked(0)

    def load2_clicked(self):
        self.load_clicked(1)

    def match_clicked(self):
        if self.matched:
            return

        if not (self.image_loaded[0] and self.image_loaded[1]):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Please load two images')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        self.do_magic()

    def draw_box(self, n, box, color, style, num):
        x1, y1, x2, y2 = box.left(), box.top(), box.right(), box.bottom()

        x1 = int(x1 * self.image_scale[n])
        y1 = int(y1 * self.image_scale[n])
        x2 = int(x2 * self.image_scale[n])
        y2 = int(y2 * self.image_scale[n])

        width = BASE_WIDTH
        if style == 'match':
            width *= 2

        painter = QPainter(self.image_pixmap[n])
        painter.setPen(QPen(QBrush(color), width))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.setPen(QPen(QBrush(TEXT_COLOR), TEXT_WIDTH))
        painter.setFont(TEXT_FONT)
        painter.drawText(x1, y2 + TEXT_SIZE + 2 * BASE_WIDTH, '{}: {}'.format(n + 1, num))
        painter.end()
        self.image_label[n].setPixmap(self.image_pixmap[n])

    def do_magic(self):
        faces_0 = self.fv.process_image(self.image_data[0])
        faces_1 = self.fv.process_image(self.image_data[1])

        n_faces_0 = len(faces_0)
        n_faces_1 = len(faces_1)

        if n_faces_0 == 0:
            show_error('No faces found on the first image')
            return

        if n_faces_1 == 0:
            show_error('No faces found on the second image')
            return

        print('Found {} face(s) on the first image'.format(n_faces_0))
        print('Found {} face(s) on the second image'.format(n_faces_1))

        rects_0 = list(map(lambda p: p[0], faces_0))
        rects_1 = list(map(lambda p: p[0], faces_1))

        embs_0 = list(map(lambda p: p[1], faces_0))
        embs_1 = list(map(lambda p: p[1], faces_1))

        scores, comps = self.fv.compare_many(self.dist, embs_0, embs_1)

        drawn_1 = [False] * n_faces_1

        for i in range(n_faces_0):
            color = BASE_COLOR
            style = 'base'

            k = np.argmax(scores[i]).item()
            if comps[i, k]:
                color = random_color()
                style = 'match'
                drawn_1[k] = True
                self.draw_box(1, rects_1[k], color, style, k)

            self.draw_box(0, rects_0[i], color, style, i)

        color = BASE_COLOR
        for j in range(n_faces_1):
            if not drawn_1[j]:
                self.draw_box(1, rects_1[j], color, 'base', j)

        tbl = TablePopup(scores, comps)
        tbl.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parser = argparse.ArgumentParser(description='Face identification using CNN + TPE')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./model',
        dest='path',
        help='path to the model'
    )
    args = parser.parse_args()
    ex = Main(args.path)
    sys.exit(app.exec_())
