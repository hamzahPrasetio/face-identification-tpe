from model import FaceVerificator
from skimage import io

fv = FaceVerificator('./model')
fv.initialize_model()
t1 = io.imread('/home/egor/Downloads/1.jpg')
t2 = io.imread('/home/egor/Downloads/2.jpg')
y = fv.compare_two(0.825, t1, t2)
print(y)
