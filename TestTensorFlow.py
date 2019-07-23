import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
a = f(2, 3)
print(a)
b = numpy.allclose(f(16.3, 12.1), 28.4)
print(b)
