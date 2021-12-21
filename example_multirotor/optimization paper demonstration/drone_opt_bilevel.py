
from drone_opt import *
import time
import timeit

start = timeit.default_timer()

bilevel_optimization(loc='urban', decomp = True)


###################################################################################
stop = timeit.default_timer()
total_time = stop - start

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d hrs:%d mins:%d secs.\n" % (hours, mins, secs))


