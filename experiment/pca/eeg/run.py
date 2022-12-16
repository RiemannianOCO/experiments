import os
foldname = '/data/regular/'
# foldname choice:
# ----------------------------------------------------------------------------
# |        name         |         diameter          |    initial point       |
# ----------------------------------------------------------------------------
# |   '/data/regular/'  |     pi / (2*sqrt(K))      |   randomly generated   |
# |   '/data/inj/'      |         pi / (2)          |   same point as above  |
# |   '/data/diam/'     |     sqrt(D) * pi / (2)    |   same point as above  |
# |   '/data/inj_bd/'   |         pi / (2)          |  close to the boarder  |
# |   '/data/diam_bd/'  |     sqrt(D) * pi / (2)    |  close to the boarder  |
# ----------------------------------------------------------------------------
os.system(f"python {os.path.dirname(__file__)}/solve_OGD.py {foldname}")
os.system(f"python {os.path.dirname(__file__)}/solve_OZO.py {foldname}")
os.system(f"python {os.path.dirname(__file__)}/solve_BAN.py {foldname}")
os.system(f"python {os.path.dirname(__file__)}/solve_2-BAN.py {foldname}")