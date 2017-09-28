import numpy as np

# MAP_FEATURES    Feature mapping function for PA2.B
#    map_features(feat1, feat2) maps the two input features
#
#    to higher-order features as defined in PA2.B
# 
#    Returns a new feature array with more features
# 
#    Inputs feat1, feat2 must be the same size
# 
#  Note: this function is only valid for PA2.B, since the degree is
#  hard-coded in.
def map_features(feat1, feat2):
    degree = 6
    out = np.ones((feat1.shape[0],1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            newc = (feat1**(i-j))*(feat2**j)
            newc.shape = (len(newc),1)
            out = np.hstack([out, newc])
    return out
