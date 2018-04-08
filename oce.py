import numpy as np
np.seterr(divide='ignore', invalid='ignore')
def PartialError(g, s):
    """Patial Error function."""

    clust_g = np.unique(g)
    clust_s = np.unique(s)
    if clust_g[0] == 0:
        clust_g = clust_g[1:]
    if clust_s[0] == 0:
        clust_s = clust_s[1:]
    err = 0.0
    if clust_s.size == 0 or clust_g.size ==0:
        return 1

    for j in clust_g:
        # initializing inner summation is the formaula.
        inner = 0.0

        Aj = (g == j)
        Wj = 0.0
        Wj += np.sum(Aj,dtype=np.float_)/(np.sum(g>0))

        # calculating denominator for Wji
        Wji_den = np.sum([(np.sum(np.logical_and(Aj, s == x))!=0) * np.sum(s == x) for x in clust_s],
                         dtype=np.float_)

        # calculating inner summation
        for i in clust_s:
            # Wji
            Bi = ( s == i )
            Wji = ((np.sum(np.logical_and(Aj,Bi),dtype=np.float_)!=0) * np.sum(Bi)) / Wji_den

            inner += (np.sum(np.logical_and(Aj,Bi),dtype=np.float_) /
                      np.sum(np.logical_or(Aj,Bi),dtype=np.float_))*Wji


        # calculating outer summation
        err += (1-inner)*Wj
    if np.isnan(err):
        return 1
    return err

def oce(gtImage, sImage):
    """Object-level Consistency Error Martin Index."""
    clust_gtImage = np.unique(gtImage)
    clust_sImage = np.unique(sImage)
    score = np.min([PartialError(gtImage, sImage), PartialError(sImage, gtImage)])
    return score

# Test for oce function.
def main():
    gt = np.array([[0, 0, 0, 0],
               [0, 1, 1, 0],
               [0, 1, 1, 0],
               [0, 0, 0, 0]])

    im1=np.array([ [0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]])

    im2=np.array([ [0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 2, 2, 0],
                   [0, 0, 0, 0]])

    im3=np.array([ [0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 2, 3, 0],
                   [0, 0, 0, 0]])
    im4=np.zeros((4,4), dtype=np.int16)

    im5=np.array([ [0, 0, 0, 0],
                   [5, 0, 0, 5],
                   [5, 0, 0, 5],
                   [0, 0, 0, 0]])

    # Test 1: Perfect match
    assert oce(gt,im1)==0

    # Test 2: Terrible match
    assert oce(gt,im4)==1

    # Test 3: Example 1
    assert oce(gt,im2)==0.5

    # Test 4: Example 2
    assert oce(gt,im3)==0.625

    # Test 5: Example 3
    assert oce(gt,im5)==1
    print("All Tests ran sucessfully.")

if __name__ == '__main__':
    main()
