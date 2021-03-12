class Clipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'lam'):
            lam = module.lam.data
            lam = lam.clamp(0.0,1)